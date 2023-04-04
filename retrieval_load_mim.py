import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import time
import datetime
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from constants import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, AutoModel

import utils
from dataset.dataset import RetrievalDataset
from models.modeling_medclip import MedCLIPVisionModelViT

from models.xbert import BertForMaskedLM
from models.tokenization_bert import BertTokenizer
class FGVLA(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        # self.medclip = MedCLIPModel(MedCLIPVisionModelViT)
        self.visual_encoder = MedCLIPVisionModelViT()
        self.vision_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(768, 512)
        self.text_encoder = AutoModel.from_pretrained(text_encoder, output_hidden_states=True)
        checkpoint = torch.load("/root/ljm/MedCLIP-main/pytorch_model.bin")
        fixed_ckpt_dict = {}
        for k, v in checkpoint.items():
            new_key = k.split("vision_model.")[-1]
            fixed_ckpt_dict[new_key] = v
        ckpt_dict = fixed_ckpt_dict
        msg = self.visual_encoder.load_state_dict(ckpt_dict, strict=False)
        print(msg)
        # self.ml_mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        # trunc_normal_(self.ml_mask_token, mean=0., std=.02)
        bert_config = BertConfig.from_json_file(config['bert_config'])
        # 768
        # 768
        text_width = self.text_encoder.config.hidden_size
        self.embed_dim = config["embed_dim"]
        # （768,512）
        self.itm_head = nn.Linear(text_width, 2)
        self.multimodel_tencoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.multimodel_vencoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / config['temp'])))

    def compute_logits(self, img_embeds, text_embeds):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(img_embeds, text_embeds.t()) * logit_scale
        return logits_per_image
@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()
    texts = data_loader.dataset.text
    num = len(texts)
    text_bs = 128
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num, text_bs):
        text = texts[i: min(num, i + text_bs)]
        # text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors="pt").to(
            device)
        output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask)
        text_embed = output.last_hidden_state
        text_feat = F.normalize(model.text_proj(text_embed[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image in data_loader:
        image = image.to(device)
        image_embed, image_feat = model.visual_encoder(image)
        img_feat = F.normalize(model.vision_proj(image_feat), dim=-1)
        image_feats.append(img_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    # 计算相似度矩阵 图片个数*num_text
    # sims_matrix = image_feats @ text_feats.t()
    sims_matrix = model.compute_logits(image_feats, text_feats)
    #  图片个数*num_text 用来存放匹配分数
    score_matrix_i2t = torch.full((num, num), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    # 找与图片最相似的topk个文本，输入fusion模块，做二分类，用 score_matrix_i2t矩阵存放score
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=(config['k_test']), dim=0)

        encoder_output = image_embeds[start + i].repeat(config['k_test'], 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        t_output = model.multimodel_tencoder.bert(encoder_embeds=text_embeds[topk_idx],
                                                  attention_mask=text_atts[topk_idx],
                                                  encoder_hidden_states=encoder_output,
                                                  encoder_attention_mask=encoder_att,
                                                  return_dict=True,
                                                  mode='fusion'
                                                  )
        v_output = model.multimodel_vencoder.bert(encoder_embeds=encoder_output,
                                                  attention_mask=encoder_att,
                                                  encoder_hidden_states=text_embeds[topk_idx],
                                                  encoder_attention_mask=text_atts[topk_idx],
                                                  return_dict=True,
                                                  mode='fusion'
                                                  )
        output = t_output.last_hidden_state[:, 0, :] * v_output.last_hidden_state[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    #
    return sims_matrix.detach().cpu().numpy(), score_matrix_i2t.detach().cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, data_loader):
    # Image->Text
    top1 = 0
    top2 = 0
    top5 = 0
    top10 = 0
    for index, score in enumerate(scores_i2t):
        # np.argsort将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
        # [::-1]取从后向前（相反）的元素
        inds = np.argsort(score)[::-1]
        # inds 现在为score从大到小排序对应的index
        cls = np.array(data_loader.dataset.label)[inds]
        cls1 = cls[:1]
        cls2 = cls[:2]
        cls5 = cls[:5]
        cls10 = cls[:10]
        top1 += len(np.where(cls1 == data_loader.dataset.label[index])[0])
        top2 += len(np.where(cls2 == data_loader.dataset.label[index])[0])
        top5 += len(np.where(cls5 == data_loader.dataset.label[index])[0])
        top10 += len(np.where(cls10 == data_loader.dataset.label[index])[0])
    # @top 1
    ir1 = 100.0 * top1 / scores_i2t.shape[0]
    # @top 2
    ir2 = 100.0 * top2 / scores_i2t.shape[0] / 2
    # @top 5
    ir5 = 100.0 * top5 / scores_i2t.shape[0] / 5
    # @top 10
    ir10 = 100.0 * top10 / scores_i2t.shape[0] / 10
    eval_result = {
        'img_r1': ir1,
        'img_r2': ir2,
        'img_r5': ir5,
        'img_r10': ir10,
    }
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    #### Dataset ####
    print("Creating retrieval dataset")
    # test_dataset = create_dataset('re', config)
    test_dataset = RetrievalDataset(
        filename='/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/mimic_5x200_sentence_-1_frontal.csv',
        imgtransform=None)

    test_loader = DataLoader(test_dataset,
                             batch_size=64,
                             collate_fn=None,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=4,
                             )


    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    #### Model ####
    print("Creating model")
    model = FGVLA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model.to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)

        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    sims_matrix, score_test_i2t = evaluation(model, test_loader, tokenizer, device, config)
    if utils.is_main_process():

        test_result = itm_eval(sims_matrix, test_loader)
        test_result2 = itm_eval(score_test_i2t, test_loader)
        print(test_result)
        print(test_result2)
        if args.evaluate:
            log_stats = {
                **{f'test1_{k}': v for k, v in test_result.items()},
                **{f'test2_{k}': v for k, v in test_result2.items()},
            }
            with open(os.path.join(args.output_dir, "evaluation.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval.yaml')
    parser.add_argument('--output_dir', default='/root/ljm/ALBEF-main/swin+cxr_bert/mim/mae/load_mim_hard/')
    parser.add_argument('--checkpoint', default="/root/ljm/ALBEF-main/swin+cxr_bert/mim/mae/load_mim_hard/checkpoint_07.pth")
    # 'microsoft/BiomedVLP-CXR-BERT-general'
    parser.add_argument('--text_encoder', default='microsoft/BiomedVLP-CXR-BERT-general')
    # 直接运行python retrieval.py，输出结果False
    # 运行python retrieval.py --evaluate，输出结果True
    parser.add_argument('--evaluate', default=True, action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)