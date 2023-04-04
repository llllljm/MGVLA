import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import utils


import torch.backends.cudnn as cudnn

from constants import constants
from dataset.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator
from models.fgvla_load_mim import FGVLA

from models.tokenization_bert import BertTokenizer

from dataset import create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import torch




# , scaler
def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, scaler):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    # metric_logger.add_meter('acc', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_sma', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
        # , annotations
    for i, inputs in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        image = inputs['pixel_values']
        img_label = inputs['img_labels']
        label = inputs['text_labels']
        text_list = inputs['text_inputs']
        aug_text_list = inputs['aug_text_inputs']
        annotation_list = inputs['annotations']
        text_input = tokenizer(text_list, padding='longest', truncation=True, max_length=77, return_tensors="pt").to(
            device)
        aug_text_input = tokenizer(aug_text_list, padding='longest', truncation=True, max_length=77, return_tensors="pt").to(
            device)
        annotations = tokenizer(annotation_list, padding='longest', truncation=True, max_length=15, return_tensors="pt").to(
            device)
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        img_label = img_label.to(device, non_blocking=True)
        with autocast():
            #  loss_ita,, loss_mim
            loss_sma, loss_mlm, loss_itm, loss_mim = model(annotations, image, img_label, text_input, label, aug_text_input)
            loss = loss_sma + loss_mlm + loss_itm + loss_mim
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        scaler.step(optimizer)
        scaler.update()
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_mim=loss_mim.item())
        metric_logger.update(loss_sma=loss_sma.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    # 初始化分布式训练 也可能不使用分布式，根据环境决定
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # start_epoch = 0
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_steps']

    #### Dataset ####
    print("Creating dataset")
    filename = '/root/datasets/mimic/files/mimic-cxr-jpg/2.0.0/img_train_na.csv'
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])],
    )

    traindata = ImageTextContrastiveDataset(filename=filename, imgtransform=transform)
    train_collate_fn = ImageTextContrastiveCollator()
    samplers = [None]
    data_loader = \
        create_loader([traindata], samplers, batch_size=[config['batch_size']], num_workers=[16], is_trains=[True],
                      collate_fns=[train_collate_fn])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    print("Creating model")
    model = FGVLA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model.to(device)
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    scaler = GradScaler()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        msg = model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config,
                            scaler)

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default="")
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--output_dir', default="/root/ljm/ALBEF-main/swin+cxr_bert/patch/")
    parser.add_argument('--text_encoder', default='microsoft/BiomedVLP-CXR-BERT-general')
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
