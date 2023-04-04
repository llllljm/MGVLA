import math
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from ruamel import yaml
from scipy import ndimage
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from transformers import AutoModel

from models.swin_transformer import SwinTransformer
from models.xbert import BertConfig, BertModel, BertForMaskedLM
from models.tokenization_bert import BertTokenizer
from models.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
import re

target_folder = "/root/datasets/ms-cxr/mask"
file = "/root/datasets/ms-cxr/cxr"
# target_folder = "/mnt/ljm/ms-cxr/mask"
# file = "/mnt/ljm/ms-cxr/cxr"
result = {"Atelectasis": [], "Cardiomegaly": [], "Consolidation": [], "Lung Opacity": [], "Edema": [], "Pneumonia": [],
          "Pneumothorax": [], "Pleural Effusion": []}
model_path = "/root/ljm/ALBEF-main/swin+cxr_bert/patch/checkpoint_09.pth"
# model_path = "/root/ljm/ALBEF-main/swin+cxr_bert/mlm/new/checkpoint_07.pth"
# _mim(feature+cls)
# model_path = "/mnt/ljm/ALBEF-mask/swin+cxr_bert/mim/checkpoint_08.pth"
config_path = "./configs/Pretrain.yaml"

transform1 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
])



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


    # def forward(self, image, text):
    #     # image_embeds, image_feats = self.medclip.visual_encoder(image)
    #     # text_output = self.medclip.text_encoder.bert(text.input_ids, text.attention_mask, mode='text')
    #     # text_embeds = text_output.last_hidden_state
    #     image_feats, image_embeds = self.visual_encoder(image)
    #     text_feats, text_embeds = self.text_em(text.input_ids, text.attention_mask)
    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    #
    #     output = self.multimodal_encoder.bert(encoder_embeds=text_embeds,
    #                                           attention_mask=text.attention_mask,
    #                                           encoder_hidden_states=image_embeds,
    #                                           encoder_attention_mask=image_atts,
    #                                           return_dict=True,
    #                                           mode='fusion'
    #                                           )
    #
    #     vl_embeddings = output.last_hidden_state[:, 0, :]
    #     vl_output = self.itm_head(vl_embeddings)
    #     return vl_output


# #  Text Preprocessing


def pre_caption(caption, max_words=100):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


device = "cuda:0"
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
model = FGVLA(text_encoder='microsoft/BiomedVLP-CXR-BERT-general', config=config)
model = model.to(device)
checkpoint = torch.load(model_path, map_location='cpu')
msg = model.load_state_dict(checkpoint["model"], strict=False)
model.eval()


# block_num = 4
#
# # model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
# model.multimodal_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True


def convert_similarity_to_image_size(
        similarity_map, width, height,
        interpolation: str = "nearest"):
    """
    Convert similarity map from raw patch grid to original image size,
    taking into account whether the image has been resized and/or cropped prior to entering the network.
    """
    n_patches_h, n_patches_w = similarity_map.shape[0], similarity_map.shape[1]
    target_shape = 1, 1, n_patches_h, n_patches_w
    # smallest_dimension = min(height, width)

    # TODO:
    # verify_resize_params(val_img_transforms, resize_size, crop_size)

    reshaped_similarity = similarity_map.reshape(target_shape)
    align_corners_modes = "linear", "bilinear", "bicubic", "trilinear"
    align_corners = False if interpolation in align_corners_modes else None
    similarity_map = F.interpolate(
        reshaped_similarity,
        size=(height, width),
        mode=interpolation,
        align_corners=align_corners,
    )[0, 0]
    return similarity_map


def cal_cnr(row):
    category_name = row["category_name"]
    width = row["image_width"]
    height = row["image_height"]
    path = row["path"]
    id = row["dicom_id"]
    image_path = os.path.join(file, id)
    image_path = image_path + ".png"
    # image_path = os.path.join(file, image_path)
    image = Image.open(image_path).convert("RGB")
    image = transform1(image).unsqueeze(0).to(device=device)
    text = row["label_text"]
    mask_path = os.path.join(target_folder, id + f'_{category_name}.png')
    mask_image = Image.open(mask_path).convert("1")
    mask_image = transform2(mask_image).squeeze(0).squeeze(0)
    mask_image = mask_image.to(device)
    mask = mask_image.flatten(0)
    ## process data
    text = pre_caption(text)
    text_input = tokenizer(text, max_length=77, return_tensors="pt").to(device=device)

    with torch.no_grad():
        # get attention
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask)
        text_embeds = text_output.last_hidden_state
        image_embeds, image_feats = model.visual_encoder(image)
        # image_feats = image_feats.unsqueeze(1)
        # image_embeds = torch.cat([image_feats, image_embeds], dim=1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        output_t = model.multimodel_tencoder.bert(encoder_embeds=text_embeds,
                                                  attention_mask=text_input.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_atts,
                                                  return_dict=True,
                                                  mode='fusion'
                                                  )
        output_v = model.multimodel_vencoder.bert(encoder_embeds=image_embeds,
                                                  attention_mask=image_atts,
                                                  encoder_hidden_states=text_embeds,
                                                  encoder_attention_mask=text_input.attention_mask,
                                                  return_dict=True,
                                                  mode='fusion',
                                                  )
        # output_t = output_t.last_hidden_state[:, 0, :].squeeze(0)
        output_t = output_t.last_hidden_state.mean(1)
        output_v = output_v.last_hidden_state[:, 1:, :].squeeze(0)
        # attention = F.cosine_similarity(output_t, output_v)
        text = F.normalize(output_t, p=2, dim=-1)
        img = F.normalize(output_v, p=2, dim=-1)
        attention = text @ img.T
        attention = attention.reshape((7, 7)).cpu().numpy()
        smoothed_attention = torch.tensor(ndimage.gaussian_filter(
            attention, sigma=(1.5, 1.5), order=0))
        attention = convert_similarity_to_image_size(smoothed_attention, width, height)
        inum = len(torch.where(mask == 1)[0])
        enum = width * height - inum
        ## interior mean and std
        attention = attention.to(device)
        interior = mask_image * attention
        interior_sum = interior.sum()
        interior_mean = interior_sum / inum
        interior_std = (interior - interior_mean)
        interior_std = interior_std * mask_image
        interior_std = interior_std * interior_std
        interior_std = interior_std.sum() / inum
        ## exterior mean and std
        mask_image = 1 - mask_image
        exterior = mask_image * attention
        exterior_sum = exterior.sum()
        exterior_mean = exterior_sum / enum
        exterior_std = (exterior - exterior_mean)
        exterior_std = exterior_std * mask_image
        exterior_std = exterior_std * exterior_std
        exterior_std = exterior_std.sum() / enum
        cnr = torch.abs(interior_mean - exterior_mean) / torch.sqrt(interior_std + exterior_std)
        if not math.isnan(cnr):
            result[category_name].append(cnr)


if __name__ == '__main__':
    filename = "/root/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.0.0.csv"
    # filename = "/mnt/ljm/ms-cxr/MS_CXR_Local_Alignment_v1.0.0.csv"
    df = pd.read_csv(filename)
    # df["boxes"] = df.apply(lambda x: create_bbox(x), axis=1)
    # df = df[["dicom_id", "category_name", "boxes", "image_width", "image_height"]]
    # # aggregate multiple boxes
    # df = df.groupby(["dicom_id", "category_name"], as_index=False).agg(list)
    # df.apply(lambda x: creat_annotation_mask(x), axis=1)
    df.apply(lambda x: cal_cnr(x), axis=1)
    dict = {"Atelectasis": 0, "Cardiomegaly": 0, "Consolidation": 0, "Lung Opacity": 0, "Edema": 0, "Pneumonia": 0,
            "Pneumothorax": 0, "Pleural Effusion": 0}
    for k, v in result.items():
        sum = 0
        for index, cnr in enumerate(v):
            sum += cnr
        avr = sum / len(v)
        dict[k] = avr
    avr = 0
    for k, v in dict.items():
        avr += v
    avr /= 8
    dict["avr"] = avr
    print(dict)
