#!/usr/bin/env python
# coding: utf-8

# # This is a notebook that shows how to produce Grad-CAM visualizations for ALBEF

# # 1. Set the paths for model checkpoint and configuration

# In[37]:
from ruamel import yaml

from models.model import MedCLIP
from models.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from models.simmim import SwinTransformerForSimMIM

model_path = "/root/ljm/ALBEF-main//Pretrain_CXR-BERT25.2/checkpoint_09.pth"
bert_config_path = 'configs/config_bertv.json'
use_cuda = True

# # 2. Model defination

# In[38]:


from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertForMaskedLM
from models.tokenization_bert import BertTokenizer

import torch
from torch import nn
from torchvision import transforms

import json


class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert=''
                 ):
        super().__init__()

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

    def forward(self, image, text):
        image_embeds, image_feats = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_feats))
        # image_atts:[128,50]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask)
        text_embeds = output.last_hidden_state
        output_t = self.multimodel_tencoder.bert(encoder_embeds=text_embeds,
                                                    attention_mask=text.attention_mask,
                                                    encoder_hidden_states=image_embeds,
                                                    encoder_attention_mask=image_atts,
                                                    return_dict=True,
                                                    mode='fusion'
                                                    )
        output_v = self.multimodel_vencoder.bert(encoder_embeds=image_embeds,
                                                    attention_mask=image_atts,
                                                    encoder_hidden_states=text_embeds,
                                                    encoder_attention_mask=text.attention_mask,
                                                    return_dict=True,
                                                    mode='fusion',
                                                    )
        output = output_t.last_hidden_state[:, 0, :] * output_v.last_hidden_state[:, 0, :]
        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output


# # 3. Text Preprocessing

# In[39]:


import re


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


# # 4. Image Preprocessing and Postpressing

# In[40]:


from PIL import Image, ImageChops
import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import gaussian_filter
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
    if blur:
        attMap = gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        ##
        # if attMap.max() > 0:
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + (attMap ** 0.7).reshape(
            attMap.shape + (1,)) * attMapV
    return attMap


normalize = transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])

# # 5. Load model and tokenizer

# In[41]:

# microsoft/BiomedVLP-CXR-BERT-general emilyalsentzer/Bio_ClinicalBERT
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")

model = VL_Transformer_ITM(text_encoder='microsoft/BiomedVLP-CXR-BERT-general', config_bert=bert_config_path)

checkpoint = torch.load(model_path, map_location='cpu')
msg = model.load_state_dict(checkpoint["model"], strict=False)
model.eval()
block_num = 5

# model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
model.multimodel_tencoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
if use_cuda:
    model.cuda()

# # 6. Load Image and Text

# In[42]:

# image_path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/p10/p10432862/s50608079/3eb2c7f8-1c178075-437d2500-02dbf9a2-c8d83b59.jpg"
# image_path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/p10/p10277852/s53356197/33dc9098-a2de042b-9e5d3472-98ce0af2-f0a8147a.jpg"
# image_path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/p19/p19206480/s52594758/d1935e38-bf72f688-646772e8-395c6a5f-1cdfc7a6.jpg"
# image_path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/p12/p12870544/s50074708/4037e95b-93ffbb99-fd878e5e-a54e0f1a-9d890e74.jpg"
# image_path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/p17/p17139582/s53630228/1a5a5eff-18da2ab4-8f020f27-e28df74a-5c2b42eb.jpg"
# image_path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/p10/p10167784/s51240614/51d2a2d2-6ae66ee2-c1633291-05a5a8e4-823dd257.jpg"
image_path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/p11/p11200955/s59417241/b05a9b0f-22529b25-d745a49b-2e1e3588-d712809d.jpg"
image_pil = Image.open(image_path).convert("RGB")
## 去黑边
# bg = Image.new(image_pil.mode, image_pil.size, image_pil.getpixel((0, 0)))
# diff = ImageChops.difference(image_pil, bg)
# bbox = diff.getbbox()
# image = image_pil.crop(bbox)
image = transform(image).unsqueeze(0)

# caption = "left basilar opacification"
# caption = "multifocal pneumonia"
# caption = "right basilar pulmonary opacity"

# caption = "the heart is mildly enlarged"
# caption = "small left apical pneumothorax"
# caption = "area of opacification has developed in the right juxta hilar region"
caption = "There is mild left basal atelectasis."
text = pre_caption(caption)
text_input = tokenizer(text, max_length=77, return_tensors="pt")

if use_cuda:
    image = image.cuda()
    text_input = text_input.to(image.device)

# # 7. Compute GradCAM

# In[43]:


output = model(image, text_input)
loss = output[:, 1].sum()

model.zero_grad()
loss.backward()

with torch.no_grad():
    mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1)
    grads = model.multimodel_tencoder.base_model.base_model.encoder.layer[
        block_num].crossattention.self.get_attn_gradients()
    cams = model.multimodel_tencoder.base_model.base_model.encoder.layer[
        block_num].crossattention.self.get_attention_map()
    #  cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
    cams = cams.reshape(image.size(0), 12, -1, 7, 7) * mask
    # 将小于0的截断为0
    # grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask
    grads = grads.clamp(0).reshape(image.size(0), 12, -1, 7, 7) * mask

    gradcam = cams * grads
    # a = gradcam[0, 0, 1, :, :]
    # gradcam[0] 12*46*24*24 mean(0)对维度为6的取平均
    # gradcam 46*24*24
    gradcam = gradcam[0].mean(0).cpu().detach()
    # b = gradcam[1, :, :]

# # 8. Visualize GradCam for each word

# In[ ]:


num_image = len(text_input.input_ids[0])
fig, ax = plt.subplots(2, 1, figsize=(15, 15))
# , figsize=(15, 15 * num_image)
# rgb_image = cv2.imread(image_path)[:, :, ::-1]
rgb_image = cv2.imread(image_path)[:, :, ::-1]
rgb_image = np.float32(rgb_image) / 255

ax[0].imshow(rgb_image)
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[0].set_xlabel("Image")
#
# for i, token_id in enumerate(text_input.input_ids[0][1:]):
#     word = tokenizer.decode([token_id])
#     # gradcam_image = getAttMap(rgb_image, gradcam[i])
#     gradcam_image = getAttMap(rgb_image, gradcam[i, :, :])
#     ax[i + 1].imshow(gradcam_image)
#     ax[i + 1].set_yticks([])
#     ax[i + 1].set_xticks([])
#     ax[i + 1].set_xlabel(word)

gradcam_image = getAttMap(rgb_image, gradcam[0, :, :])
ax[1].imshow(gradcam_image)
ax[1].set_yticks([])
ax[1].set_xticks([])

plt.savefig('/root/ljm/ALBEF-main/swin+cxr_bert/mim/mae//There is mild left basal atelectasis.png')
# plt.show()
print("over")
