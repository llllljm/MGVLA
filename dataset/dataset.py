import json
import os
import re
import random
from collections import defaultdict
import pdb
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms

from transformers import AutoTokenizer
from transformers import CLIPFeatureExtractor, CLIPProcessor
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor

# from nltk.tokenize import RegexpTokenizer
import nltk
from PIL import Image

from constants import constants


class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self,
                 do_resize=True,
                 size=224,
                 resample=Image.BICUBIC,
                 do_center_crop=True,
                 crop_size=224,
                 do_normalize=True,
                 image_mean=constants.IMG_MEAN,
                 image_std=constants.IMG_STD,
                 do_convert_rgb=False,
                 do_pad_square=True,
                 **kwargs):
        super().__init__(do_resize, size, resample, do_center_crop, crop_size, do_normalize, image_mean, image_std,
                         do_convert_rgb, **kwargs)
        self.do_pad_square = do_pad_square

    def __call__(self,
                 images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List[
                     "torch.Tensor"]],
                 return_tensors: Optional[Union[str, TensorType]] = None,
                 **kwargs) -> BatchFeature:
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (convert rgb + resizing + center cropping + normalization)
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]

        if self.do_pad_square:
            images = [self.pad_img(image, min_size=self.size) for image in images]

        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # add a RGB dim for each image
        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        images = images_

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im


class MedCLIPProcessor(CLIPProcessor):
    '''
    A processor that takes input images and texts and provides inputs for
    `MedCLIPModel`.
    '''
    feature_extractor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self):
        feature_extractor = MedCLIPFeatureExtractor()
        tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        tokenizer.model_max_length = 77
        super().__init__(feature_extractor, tokenizer)


class ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
                'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                'Fracture', 'Support Devices']

    def __init__(self, filename="/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/img_train.csv", imgtransform=None) -> None:
        # '''support data list in mimic-cxr-train, chexpert-train
        # ['mimic-cxr-train', 'chexpert-train']
        # '''
        super().__init__()
        self.df = pd.read_csv(filename)
        self.df = self.df[self.df["report"] != " "]
        self.df = self.df.fillna(-1)
        # # build dict from dataframe
        # self.df['subject_id'] = self.df['subject_id'].astype('str')
        # self.df['study_id'] = self.df['study_id'].astype('str')
        # self.df["subject_id"] = self.df['subject_id'].str.cat(self.df['study_id'], sep="/")
        # keys = self.df['subject_id'].values
        # vals = self.df.drop(['subject_id', 'study_id'], axis=1).fillna(-1).values
        # self.img_label_dict = dict(zip(keys, vals))
        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)
        # with open(filename, 'r', encoding='utf-8') as file1:
        #     self.objs = json.load(file1)
        # file1.close()
        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/sentence-label_na.csv')
        self.sentence_label = self.sentence_label[self.sentence_label["reports"] != " "]
        self.sentence_label = self.sentence_label.fillna(-1)
        print('load sentence prompts from /root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/sentence-label_na.csv')
        self._preprocess_sentence_label()
        # self.objs = self.create_sent_segments(self.objs)
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        reports = row.report
        path = "/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/files/"
        subject = row["subject_id"].astype('str')
        path = path + "p" + subject[:2]
        path = os.path.join(path, "p" + subject)
        study = row["study_id"].astype('str')
        path = os.path.join(path, "s" + study)
        dicom = row["dicom_id"]
        path = os.path.join(path, dicom + ".jpg")
        img = Image.open(path)
        img = self._pad_img(img)  # pad image to square
        img = self.transform(img).unsqueeze(1)
        img_label = row[self._labels_].values # image corresponds to text labels
        if len(reports) == 0: # no report available
            # sample class prompts as augmentation
            report, text_label = self.sample_sent_prompts(row)
            annotation = " "
        else:
            #  sample one sentence according to weights
            sent_ix = 0
            weights = torch.zeros(len(reports))
            for i, report in enumerate(reports):
                if report in self.sent_label_dict:
                    weights[i] = self.sent_weight_dict[report]
            if len(torch.where(weights == 0)[0]) < len(weights):
                sent_ix = torch.multinomial(weights, 1).item()
            else:
                sent_ix = random.randint(0, len(reports)-1)
            # sent_ix = random.randint(0, len(report)-1)
            report = reports[sent_ix]
            # we need to use sentence-level label instead
            # maintain a sentence dictionary
            # index sentence dictionary label during training, if not found, return all zero
            # **supervision from unpaired texts and images**
            if report in self.sent_label_dict: # find the sentence
                text_label = self.sent_label_dict[report]
                annotation = self.sent_annotation_dict[report]

            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1
                annotation = " "


        return img, report, img_label, text_label, annotation


    def __len__(self):
        # return len(self.objs)
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all():  # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] * row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:, 1:] == 0).all(1)]
            if len(sents) == 0:  # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding'] == 1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n', ' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            # reports = [point.split(". ") for point in report]
            reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = nltk.wordpunct_tokenize(sent.lower())

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4:  # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='reports')
        self.sentence_label = self.sentence_label[self.sentence_label['reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['reports']
        self.sentence_label = self.sentence_label.drop('reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        annotations_vals = self.sentence_label['annotations'].values
        weights_vals = self.sentence_label['weights'].values
        vals = self.sentence_label.drop(['report', 'annotations', 'weights'], axis=1).values
        self.sent_label_dict = dict(zip(keys, vals))
        self.sent_annotation_dict = dict(zip(keys, annotations_vals))
        self.sent_weight_dict = dict(zip(keys, weights_vals))

    def _build_prompt_sentence(self, n=200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n:
                new_sent_list.append(sub_sent_df)
            else:
                new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, 0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df


class ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            # import nltk
            # nltk.download('stopwords')
            # nltk.download('omw-1.4')
            # nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77

    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        annotation_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0])
            if self.eda is not None:
                eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                text_aug = eda_aug(data[1])
                if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                report_aug_list.append(text_aug)
            report_list.append(data[1])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
            annotation_list.append(data[4])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        # annotations = self.tokenizer(annotation_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1, 3, 1, 1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['text_inputs'] = report_list
        inputs['annotations'] = annotation_list
        if len(report_aug_list) > 0:
            inputs['aug_text_inputs'] = report_aug_list

        return inputs

class RetrievalDataset(Dataset):
    def __init__(self,
                 filename='/mnt/ljm/ALBEF-mask/mimic_5x200_test.csv',
                 imgtransform=None,
                 ) -> None:
        '''support data list in mimic-5x200, chexpert-5x200, rsna-balanced-test, covid-test
        args:
            imgtransform: a torchvision transform
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform
        self.df = pd.read_csv(filename)
        self.text = []
        # self.images = []
        self.label = []
        # self.images_label = []
        for i in range(0, 1000):
            self.text.append(self.df.iloc[i]['Report'])
            label = self.df.iloc[i][constants.CHEXPERT_COMPETITION_TASKS].values.flatten()
            for j in range(len(constants.CHEXPERT_COMPETITION_TASKS)):
                if label[j] == 1:
                    self.label.append(j)
                    continue
            # for j in range(0, len(constants.CHEXPERT_TASKS)):
            #     if self.df.iloc[i][constants.CHEXPERT_TASKS[j]] == 1:
            #         self.label.append(constants.CHEXPERT_COMPETITION_TASKS[constants.CHEXPERT_TASKS[j]])
            #         break

    def __len__(self):
        return 1000
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # path = row['Path'].split("/root/datasets")[-1]
        # path = "/mnt/ljm" + path
        img = Image.open(row['Path'])
        img = self._pad_img(img)
        img = self.transform(img)
        # label = self.label[index]
        if img.shape[0] == 1: img = img.repeat((3, 1, 1))
        return img

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)
class SuperviseImageDataset(Dataset):
    def __init__(self,
                 filename,
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        super().__init__()
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names
        self.df = pd.read_csv(filename)
        self.df = self.df.sample(frac=0.1, random_state=60)


    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row['Path'])
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class SuperviseImageCollator:
    def __init__(self, mode):
        assert mode in ['multiclass', 'multilabel', 'binary']
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])
        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values

        if self.mode in ['multiclass', 'binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1, 3, 1, 1))
        return {
            'pixel_values': inputs['pixel_values'],
            'labels': inputs['labels'],
        }


