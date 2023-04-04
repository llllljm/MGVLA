from functools import partial

import numpy as np
import torch
from ruamel import yaml

from torch import nn
from transformers import AutoModel
from timm.models.layers import trunc_normal_

from models.modeling_medclip import MedCLIPVisionModelViT

from models.xbert import BertConfig, BertForMaskedLM
import torch.nn.functional as F
from constants.constants import *
import random


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
        # self.decoder = nn.Sequential(
        #         nn.Conv2d(
        #             in_channels=self.visual_encoder.num_features,
        #             out_channels=32 ** 2 * 3, kernel_size=1),
        #         nn.PixelShuffle(32),
        #     )
        self.ml_mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        trunc_normal_(self.ml_mask_token, mean=0., std=.02)
        self.ml_cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        trunc_normal_(self.ml_cls_token, mean=0., std=.02)
        self.vclassify = nn.Linear(768, 14 * 3)
        # self.tclassify = nn.Linear(768, 14 * 3)
        self.decoder = nn.Linear(768, 768)

    def forward(self, annotations, image, img_label, text, label, aug_text):
        # image_embeds：[128,49,768]
        # image_feats:[128,512]
        image_embeds, image_feats = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_feats))
        # image_atts:[128,50]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask)
        text_embeds = output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]))
        aug_text_output = self.text_encoder(aug_text.input_ids, attention_mask=aug_text.attention_mask)
        aug_text_embeds = aug_text_output.last_hidden_state
        aug_text_feat = F.normalize(self.text_proj(aug_text_embeds[:, 0, :]))
        # ================= sml ========================##
        # compute soft-labels, -1: negative, 0: uncertain, 1: positive
        label_sim = torch.matmul(img_label, label.T)
        label_sim = label_sim.to(image.device)

        sim_i2t = self.compute_logits(image_feat, text_feat)
        #
        loss_value = self.soft_clip_loss(sim_i2t, label_sim)
        logits_aug = self.compute_logits(image_feat, aug_text_feat)
        aug_loss_value = self.soft_clip_loss(logits_aug, label_sim)
        #
        loss_sma = (loss_value + aug_loss_value) / 2
        # ###================itm=================###
        # save attention map for mim
        self.multimodel_tencoder.base_model.base_model.encoder.layer[4].crossattention.self.save_attention = True
        # forward the positve image-text pair
        output_tpos = self.multimodel_tencoder.bert(encoder_embeds=text_embeds,
                                                    attention_mask=text.attention_mask,
                                                    encoder_hidden_states=image_embeds,
                                                    encoder_attention_mask=image_atts,
                                                    return_dict=True,
                                                    mode='fusion'
                                                    )
        attention_map = self.multimodel_tencoder.base_model.base_model.encoder.layer[
            4].crossattention.self.get_attention_map()
        output_vpos = self.multimodel_vencoder.bert(encoder_embeds=image_embeds,
                                                    attention_mask=image_atts,
                                                    encoder_hidden_states=text_embeds,
                                                    encoder_attention_mask=text.attention_mask,
                                                    return_dict=True,
                                                    mode='fusion',
                                                    )
        output_pos = output_tpos.last_hidden_state[:, 0, :] * output_vpos.last_hidden_state[:, 0, :]
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(label_sim, dim=1)
            weights_t2i = F.softmax(label_sim.T, dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)
        #
        # # select a negative image for each text
        image_embeds_neg = []
        # label_tneg = []
        for b in range(bs):
            # choose a hard negative(similar but label is different)
            # weights_t2i[b][weights_t2i[b] == weights_t2i[b][b]] = 0
            if len(torch.where(weights_t2i[b] == 0 )[0]) == bs:
                neg_idx = random.randint(0, bs-1)
            else :
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            # label_tneg.append(weights_t2i[b][neg_idx])
            image_embeds_neg.append(image_embeds[neg_idx])
        # label_tneg = torch.stack(label_tneg)
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        # label_ineg = []
        for b in range(bs):
            # weights_i2t[b][weights_i2t[b] == weights_i2t[b][b]] = 0
            if len(torch.where(weights_i2t[b] ==0)[0]) == bs:
                neg_idx = random.randint(0, bs-1)
            else:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            # label_ineg.append(weights_i2t[b][neg_idx])
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        # label_ineg = torch.stack(label_ineg, dim = 0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat((text.attention_mask, text_atts_neg), dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        output_tneg = self.multimodel_tencoder.bert(encoder_embeds=text_embeds_all,
                                                    attention_mask=text_atts_all,
                                                    encoder_hidden_states=image_embeds_all,
                                                    encoder_attention_mask=image_atts_all,
                                                    return_dict=True,
                                                    mode='fusion'
                                                    )
        output_vneg = self.multimodel_vencoder.bert(encoder_embeds=image_embeds_all,
                                                    attention_mask=image_atts_all,
                                                    encoder_hidden_states=text_embeds_all,
                                                    encoder_attention_mask=text_atts_all,
                                                    return_dict=True,
                                                    mode='fusion'
                                                    )
        output_neg = output_tneg.last_hidden_state[:, 0, :] * output_vneg.last_hidden_state[:, 0, :]
        vl_embeddings = torch.cat((output_pos, output_neg), dim=0)
        vl_output = self.itm_head(vl_embeddings)
        # label_neg = torch.cat([label_ineg, label_tneg], dim = 0)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        # itm_labels = torch.cat([torch.ones(bs, dtype=torch.float32).to(image.device), label_neg], dim=0)
        # loss_itm = -torch.sum(F.log_softmax(vl_output, dim=1) * itm_labels, dim=1).mean()
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= MLM ========================##

        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = self.knowledge_mask(annotations, input_ids, self.mlm_probability)
        # torch.full（）用mlm_probability填充矩阵 [[0.15,0.15...0.15]...[0.15...]] tensor(64,100)
        # probability_matrix = torch.full(labels.shape, self.mlm_probability).to(image.device)
        # 对input_ids更改
        input_ids, labels = self.mask(input_ids, self.multimodel_tencoder.config.vocab_size, image.device,
                                      targets=labels,
                                      probability_matrix=probability_matrix)
        mask_output = self.text_encoder(input_ids, text.attention_mask)
        mask_text_embeds = mask_output.last_hidden_state
        mlm_output = self.multimodel_tencoder(encoder_embeds=mask_text_embeds,
                                              attention_mask=text.attention_mask,
                                              encoder_hidden_states=image_embeds,
                                              encoder_attention_mask=image_atts,
                                              mode='fusion',
                                              return_dict=True,
                                              labels=labels,
                                              )
        loss_mlm = mlm_output.loss
        ##================= MIM ========================#
        mask_embeds = image_embeds.clone()
        img_label = mask_embeds.clone()
        with torch.no_grad():
            text_mask = text.attention_mask.view(text.attention_mask.size(0), 1, -1, 1, 1)
            attention_map = attention_map[:, :, :, 1:].reshape(image.size(0), 12, -1, 7, 7) * text_mask
            N, L, D = mask_embeds[:, 1:, :].shape  # batch, length, dim
            len_keep = int(L * (1 - 0.25))
            attention_map = attention_map.mean(1)
            attention_map = attention_map[:, 0]
            attention_map = attention_map.reshape(N, 49)
            # noise = torch.rand(N, L, device=image.device)
            ids_shuffle = torch.argsort(attention_map, dim=1).to(image.device)  # ascend: small is keep, large is remove
            # ids_shuffle = torch.argsort(noise, dim=1).to(image.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1).to(image.device)
            ids_keep = ids_shuffle[:, :len_keep]
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=image.device)  # 64*49
            mask[:, :len_keep] = 0  # 每行后49个为0
            # unshuffle to get the binary mask
            mask_patch = torch.gather(mask, dim=1, index=ids_restore)  # 64*49
            x_unmasked = torch.gather(mask_embeds[:, 1:, :], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 768))
        # don't update vision_encoder,but update multimodel_vencoder
        mask_tokens = self.ml_mask_token.repeat(x_unmasked.shape[0], ids_restore.shape[1] + 1 - x_unmasked.shape[1], 1)
        ml_cls_token = self.ml_cls_token.repeat(x_unmasked.shape[0], 1, 1)
        ml_cls_token = torch.cat((ml_cls_token, x_unmasked), dim=1)
        avgpool = nn.AdaptiveAvgPool1d(1)
        ml_cls_token = avgpool(ml_cls_token.transpose(1, 2))
        ml_cls_token = ml_cls_token.transpose(1, 2)
        mask_embeds = torch.cat([x_unmasked, mask_tokens], dim=1)  # no cls token
        mask_embeds = torch.gather(mask_embeds, dim=1,
                                   index=ids_restore.unsqueeze(-1).repeat(1, 1, x_unmasked.shape[2]))  # unshuffle
        mask_embeds = torch.cat([ml_cls_token, mask_embeds], dim=1)  # append cls token
        mim_output = self.multimodel_vencoder.bert(encoder_embeds=mask_embeds,
                                                   attention_mask=image_atts,
                                                   encoder_hidden_states=text_embeds,
                                                   encoder_attention_mask=text.attention_mask,
                                                   return_dict=True,
                                                   mode='fusion'
                                                   )
        mask_embeds = mim_output.last_hidden_state
        rec = self.decoder(mask_embeds[:, 1:, :])
        loss_recon = F.l1_loss(img_label[:, 1:, :], rec, reduction='none')
        mask_patch = mask_patch.unsqueeze(-1)
        loss_mim = (loss_recon * mask_patch).sum() / (mask_patch.sum() + 1e-5) / 768
        cls = self.vclassify(mask_embeds[:, 0, :])
        # 14*3
        cls = cls.reshape(bs * 14, 3)
        label = label + 1
        label = torch.tensor(label, dtype=torch.long)
        label = label.reshape(bs * 14)
        loss_cls = F.cross_entropy(cls, label)
        loss_mim = loss_cls + loss_mim

        return loss_sma, loss_mlm, loss_itm, loss_mim

    def compute_logits(self, img_embeds, text_embeds):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(img_embeds, text_embeds.t()) * logit_scale
        return logits_per_image

    def soft_clip_loss(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        '''
        # when using InfoNCE-like loss
        image_loss = self.soft_xent_loss(logits_per_img, F.softmax(soft_label, 1))
        caption_loss = self.soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T, 1))
        return (image_loss + caption_loss) / 2

    def soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return -(target * logprobs).sum() / input.shape[0]

    def knowledge_mask(self, annotations, input_id, mlm_probability):
        """
        Get 0/1 labels for masked tokens
        """
        mask_labels = []
        for o, e in enumerate(input_id):
            masked_lms = []
            input_tokens = []
            for i, id in enumerate(e):
                # if len(masked_lms) < input_id.shape[1] * mlm_probability:
                token = self.tokenizer._convert_id_to_token(id.item())
                input_tokens.append(token)
                if id != self.tokenizer.cls_token_id and id != self.tokenizer.pad_token_id:
                    # if id in hpo.input_ids[o]:
                    #     masked_lms.append(i)
                    if id in annotations.input_ids[o]:
                        masked_lms.append(i)
                    elif id in words_id:
                        masked_lms.append(i)

            mask_label = [1 if i in masked_lms else 0 for i in range(len(e))]
            mask_labels.append(mask_label)
        mask_labels = torch.tensor(mask_labels).cuda()
        mask_labels = mask_labels.bool()
        return mask_labels

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        # mask(input_ids, self.text_encoder.config.vocab_size, image.device,
        # targets=labels, probability_matrix = probability_matrix)
        if masked_indices is None:
            masked_indices = probability_matrix
            # masked_indices = torch.bernoulli(probability_matrix).bool()

        # 特殊词不进行mask
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            # ~ ： 按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1
            # True的位置填充-100 即masked_indices为False 不需要mask的位置填-100
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # torch.full(size, fill_value）
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8).cuda()).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5).cuda()).bool() & masked_indices & ~indices_replaced
        # torch.randint(high,size,dtype)
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def img_mask(self, attention, mim_probability=0.75, input_size=224, mask_patch_size=32, model_patch_size=4):

        rand_size = input_size // mask_patch_size
        scale = mask_patch_size // model_patch_size
        token_count = rand_size ** 2
        mask_count = int(np.ceil(token_count * mim_probability))
        attention = attention.flatten(0)
        mask_idx = torch.multinomial(attention, mask_count)
        # mask_idx = np.random.permutation(token_count)[:mask_count]

        mask = torch.zeros(token_count, dtype=int)
        mask[mask_idx] = 1
        mask1 = mask.unsqueeze(0)
        mask2 = mask.reshape((rand_size, rand_size))
        mask2 = torch.repeat_interleave(mask2, scale, dim=0)
        mask2 = torch.repeat_interleave(mask2, scale, dim=1)
        mask2 = mask2.unsqueeze(0)
        return mask1, mask2

class PromptClassifier(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''

    def __init__(self, model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        input_ids = prompt_inputs["input_ids"].cuda()
        attention_mask = prompt_inputs["attention_mask"].cuda()
        image_embed, image_feat = self.model.visual_encoder(pixel_values)
        img_feat = F.normalize(self.model.vision_proj(image_feat), dim=-1)
        output = self.model.text_encoder(input_ids, attention_mask)
        text_embed = output.last_hidden_state
        text_feat = F.normalize(self.model.text_proj(text_embed[:, 0, :]))
        new_logits = self.model.compute_logits(img_feat, text_feat)
        bs = pixel_values.size()[0]
        # logits = torch.full((bs, 50), -100.0).cuda()
        logits = torch.full((bs, 10), -100.0).cuda()
        for i in range(bs):
            # topk_sim, topk_idx = new_logits[i].topk(k=5, dim=0)
            # encoder_output = image_embed[i].unsqueeze(0).repeat(5, 1, 1)
            topk_sim, topk_idx = new_logits[i].topk(k=10, dim=0)
            encoder_output = image_embed[i].unsqueeze(0).repeat(10, 1, 1)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).cuda()
            t_output = self.model.multimodel_tencoder.bert(encoder_embeds=text_embed[topk_idx],
                                                           attention_mask=attention_mask[topk_idx],
                                                           encoder_hidden_states=encoder_output,
                                                           encoder_attention_mask=encoder_att,
                                                           return_dict=True,
                                                           mode='fusion'
                                                           )
            v_output = self.model.multimodel_vencoder.bert(encoder_embeds=encoder_output,
                                                           attention_mask=encoder_att,
                                                           encoder_hidden_states=text_embed[topk_idx],
                                                           encoder_attention_mask=attention_mask[topk_idx],
                                                           return_dict=True,
                                                           mode='fusion'
                                                           )
            output = t_output.last_hidden_state[:, 0, :] * v_output.last_hidden_state[:, 0, :]
            logit = self.model.itm_head(output)[:, 1]
            logits[i, topk_idx] = logit
        outputs = {
            'logits': logits
        }
        return outputs