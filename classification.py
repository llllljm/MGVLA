import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from constants import constants
from dataset.dataset import SuperviseImageDataset, SuperviseImageCollator
from models.classifier import Classifer
from models.fgvla_classification import SuperviseClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(arg, config):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating fine tune dataset")
    # val_data = SuperviseImageDataset(
    #     '/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/mimic_5x200_sentence_-1_frontal.csv',
    #     class_names=constants.CHEXPERT_COMPETITION_TASKS)
    val_data = SuperviseImageDataset(
        '/root/datasets/COVID-19_Radiography_Dataset/0.3covid.csv',
        class_names=constants.COVID_TASKS)
    # val_data = SuperviseImageDataset(
    #     '/root/datasets/rsna_pneumonia/0.3rsna.csv',
    #     class_names=constants.RSNA_TASKS)
    # val_collate_fn = SuperviseImageCollator(mode='multiclass')
    # val_collate_fn = SuperviseImageCollator(mode='multiclass')
    val_collate_fn = SuperviseImageCollator(mode='binary')
    eval_dataloader = DataLoader(val_data,
                                 batch_size=64,
                                 collate_fn=val_collate_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=4,
                                 )

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    # model = SuperviseClassifier(vision_model=vision_model, mode="multiclass")
    model = SuperviseClassifier(config=config, text_encoder=args.text_encoder, num_class=2, mode="binary")
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        msg = model.load_state_dict(checkpoint["model"], strict=False)

        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)

    # evaluator = Classifer(
    #     SuperviseClassifier=model,
    #     eval_dataloader=eval_dataloader,
    #     mode='multiclass',
    # )
    evaluator = Classifer(
        SuperviseClassifier=model,
        eval_dataloader=eval_dataloader,
        mode='binary',
    )
    output = evaluator.evaluate()
    # retrieval = Retrieval(model, data_loader=test_loader, tokenizer=tokenizer, device=device)
    # sims_matrix = retrieval.evaluation()
    # test_result = retrieval.calculate(sims_matrix)
    print(output["acc"])
    print(output["auc"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval.yaml')
    parser.add_argument('--output_dir', default='/root/ljm/MedCLIP-main/')
    # MedCLIP-main MedCLIP-main/checkpoints/pretrain/best/
    # parser.add_argument('--checkpoint',
    #                     default='/mnt/ljm/medclip/MedCLIP-main/checkpoints/pretrain_select/51000/pytorch_model.bin')
    # parser.add_argument('--checkpoint',
    #                     default='/root/ljm/MedCLIP-main/checkpoints/origin_data/pytorch_model.bin')
    # parser.add_argument('--checkpoint',
    #                     default='/root/ljm/MedCLIP-main/checkpoints/fine-tune/mimic/150/pytorch_model.bin')
    parser.add_argument('--checkpoint',
                        default='/root/ljm/ALBEF-main/swin+cxr_bert/finetune/0.1covid/checkpoint_09.pth')
    # parser.add_argument('--checkpoint',
    #                     default='/root/ljm/MedCLIP-main/checkpoints/fine-tune/1rsna/3000/pytorch_model.bin')
    # pretrain/54000/
    parser.add_argument('--text_encoder', default="emilyalsentzer/Bio_ClinicalBERT")
    # 直接运行python retrieval.py，输出结果False
    # 运行python retrieval.py --evaluate，输出结果True
    parser.add_argument('--evaluate', default=True, action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args,config)
