import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ruamel import yaml
from torchvision import transforms

from dataset import create_dataset, create_sampler, create_loader
from torch import nn
import torch.nn.functional as F
from models.model import MedCLIP
from models.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from models.xbert import BertConfig, BertModel, BertForMaskedLM
from models.tokenization_bert import BertTokenizer

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}
device = "cuda:0"

# class VL_Transformer_ITM(nn.Module):
#     def __init__(self,
#                  text_encoder=None,
#                  config_bert=''
#                  ):
#         super().__init__()
#
#         self.medclip = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
#         bert_config = BertConfig.from_json_file(config_bert)
#         self.multimodal_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
#         self.itm_head = nn.Linear(768, 2)
#
#     def forward(self, image, text):
#         image_feats, image_embeds = self.medclip.vision_model(image)
#         text_feats, text_embeds = self.medclip.text_model(text.input_ids, text.attention_mask)
#
#         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
#
#         output = self.multimodal_encoder.bert(encoder_embeds=text_embeds,
#                                               attention_mask=text.attention_mask,
#                                               encoder_hidden_states=image_embeds,
#                                               encoder_attention_mask=image_atts,
#                                               return_dict=True,
#                                               )
#
#         vl_embeddings = output.last_hidden_state[:, 0, :]
#         vl_output = self.itm_head(vl_embeddings)
#         return vl_output
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def generate_chexpert_class_prompts(n=None):
    """Generate text prompts for each CheXpert classification task
    Parameters
    ----------
    n:  int
        number of prompts per class
    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        # randomly sample n prompts for zero-shot classification
        # TODO: we shall make use all the candidate prompts for autoprompt tuning
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts


def zero_shot_classification(model, imgs, promt_feats, ensemble=True):
    model.eval()
    # Returns
    # -------
    # cls_similarities :
    #     similartitie between each imgs and text

    # get similarities for each class
    imgs = imgs.to(device)
    imgs = imgs.unsqueeze(0)
    class_similarities = []
    for k, v in promt_feats.items():
        num = len(v)
        class_similarity = []
        for i in range(0, num, 128):
            text_feat = v[i: min(num, i + 128)]
            image_embed, image_feat = model.visual_encoder(imgs)
            image_feat = F.normalize(model.vision_proj(image_feat), dim=-1)
            score = image_feat @ text_feat.T
            class_similarity.append(score)
        class_similarity = torch.cat(class_similarity, dim=1)
        if ensemble:
            cls_sim = torch.mean(class_similarity, 1)  # equivalent to prompt ensembling
        else:
            cls_sim = torch.max(class_similarity, 1)[0]
        class_similarities.append(cls_sim)
    class_similarities = torch.stack(class_similarities, axis=1)
    class_similarities = class_similarities.squeeze(0)
    return class_similarities

def read(file):
    df = pd.read_csv(file)
    dir = {"Atelectasis": 0,"Cardiomegaly":1,"Consolidation":2,"Edema":3, "Pleural Effusion":4}
    dir2 = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
            "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
            "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
    text = []
    images = []
    texts_label = []
    images_label = []
    for i in range(0, 1000):
        # text.append(df.iloc[i]['Report Impression'])
        text.append(df.iloc[i]['Report'])
        path = df.iloc[i]['Path'].split("/root/datasets")[-1]
        path = "/mnt/ljm" + path
        images.append(path)
        # images.append(df.iloc[i]['Path'])
        for j in range(0, len(dir2)):
            if df.iloc[i][dir2[j]] == 1:
                texts_label.append(dir[dir2[j]])
                images_label.append(dir[dir2[j]])
                break

    return text, images, texts_label, images_label

if __name__ == '__main__':
    model_path = "/mnt/ljm/ALBEF-mask/medclip/checkpoint_07.pth"
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_config_path = 'configs/config_bert.json'
    config = yaml.load(open('./configs/Retrieval.yaml', 'r'), Loader=yaml.Loader)

    model = MedCLIP(config=config, text_encoder="emilyalsentzer/Bio_ClinicalBERT", tokenizer=tokenizer)
    checkpoint = torch.load(model_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    model.cuda()

    prompts = generate_chexpert_class_prompts()
    # file = "/mnt/ljm/ALBEF-mask/chexpert_5x200.csv"
    file = "/mnt/ljm/ALBEF-mask/mimic_5x200_test.csv"
    text, images, texts_label, images_label = read(file)
    num = len(images)
    right = 0
    promt_feats = {"Atelectasis": [], "Cardiomegaly": [], "Consolidation": [], "Edema": [], "Pleural Effusion": []}
    for cls_name, cls_text in prompts.items():
        text_num = len(cls_text)
        for i in range(0, text_num, 128):
            text = cls_text[i: min(text_num, i + 128)]
            text_input = tokenizer(text, padding='longest', truncation=True, max_length=77, return_tensors="pt").to(device)
            text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask)
            text_embed = text_output.last_hidden_state
            text_feat = F.normalize(model.text_proj(text_embed[:, 0, :]), dim=-1)
            promt_feats[cls_name].append(text_feat)
        promt_feats[cls_name] = torch.cat(promt_feats[cls_name],dim=0)
    for i, image in enumerate(images):
        img = Image.open(image).convert('RGB')
        img = transform(img)
        similarity = zero_shot_classification(model, img, promt_feats)
        similarity = similarity.detach().cpu().numpy()
        inds = np.argsort(similarity)[::-1]
        label = images_label[i]
        if inds[0] == label:
            right += 1

    acc = right/num * 100
    # dataset = [create_dataset('re', config=config)]
    # samplers = [None]
    # dataloader = create_loader(dataset, samplers, batch_size=[64], num_workers=[4],
    #                            is_trains=[False],
    #                            collate_fns=[None])[0]
    #
    # for image, img_id, label in dataloader:
    #     image = image.cuda()
    #     zero_shot_classification(model, image, prompts, tokenizer)
    print(acc)
