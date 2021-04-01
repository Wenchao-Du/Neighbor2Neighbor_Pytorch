#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/04/01
@Author  :   Garified Du
@Version :   1.0
@Desc    :   Update the dist training for N2N, where replace the torch.dist with NVIDIA-apex libs
'''

# here put the import lib

# import the baselib
import os
import sys
import random
import glob
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from Utils.utils import get_config
import matplotlib.pyplot as plt

# import the third libs
import numpy as np

# import torch libs
import torch
from torch import nn
from torch.utils.data import DataLoader

# import models and metrics
from libs.unet import Unet as UNet
from libs.babyunet import BabyUnet
from libs.rednet import REDNet10, REDNet20, REDNet30
from libs.genmask import masksampling, generator, masksamplingv2

from libs.loss import define_loss, compare_psnr, compare_ssim
from Utils.utils import define_optim, define_scheduler, Logger, AverageMeter
from libs.benchmark_metrics import Metrics
from data import get as get_data

# import dist libs
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

parser = argparse.ArgumentParser(
    description='Pytorch Dist training for Neighbor2Neighbor')
parser.add_argument('--config', type=str, default='configs/yourconfig.yaml')
args = parser.parse_args()
config = get_config(args.config)
config['num_gpus'] = len(config['gpuid'].split(','))

os.environ['CUDA_VISBILE_DEVICES'] = config['gpuid']
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = config['port']

# define the random seeds
torch.manual_seed(config['seed'])
torch.cuda.manual_seed_all(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
""" map-reduce the losses 
"""


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt)
    rt /= config['num_gpus']
    return rt


"""
    @ validate the model on the testset
"""


def val(model, setloader, epoch, gpu):
    psnrs = []
    ssims = []
    model.eval()
    with torch.no_grad():
        for _, inputs in tqdm(enumerate(setloader)):
            noise = inputs['noise'].cuda()
            output = model(noise)
            # output = torch.clamp(output, min=0., max=1.)
            # cal the metric and update the avg_stc
            gt = inputs['gt'].squeeze().numpy()
            output = output.squeeze().cpu().numpy()
            psnr = compare_psnr(gt, output)
            ssim = compare_ssim(gt, output)
            psnrs.append(psnr)
            ssims.append(ssim)
    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    model.train()
    print("PSNR, SSIM:{},{}\n".format(psnrs.mean(), ssims.mean()))
    return psnrs.mean(), ssims.mean()


def train(gpu, config):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=config['num_gpus'],
                            rank=gpu)
    torch.cuda.set_device(gpu)
    """ 
        @ build the dataset for training
    """
    dataset = get_data(config)
    trainset = dataset(config, "train")
    testset = dataset(config, "test")
    sampler_train = DistributedSampler(trainset,
                                       num_replicas=config['num_gpus'],
                                       rank=gpu)
    sampler_val = DistributedSampler(testset,
                                     num_replicas=config['num_gpus'],
                                     rank=gpu)

    batch_size = config['batch_size']
    loader_train = DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=config['num_threads'],
                              pin_memory=True,
                              sampler=sampler_train,
                              drop_last=True)
    loader_val = DataLoader(dataset=testset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            sampler=sampler_val,
                            drop_last=True)
    model = UNet(config["in_channels"],
                 config["out_channels"],
                 post_processing=True)
    model.cuda(gpu)
    mask_sampling = masksamplingv2()
    """  @ init parameter
    """

    save_folder = os.path.join(
        config['save_root'], 'batch_{}_lr_{}'.format(config['batch_size'],
                                                     config['lr']))
    best_epoch = 0
    lowest_loss = 0.
    resume = 0
    print('=>Save folder: {}\n'.format(save_folder))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    optimizer = define_optim(config['optimizer'], model.parameters(),
                             float(config['lr']), 0)

    criterion_1 = define_loss(config['loss_type'])
    criterion_2 = define_loss("Multimse")
    scheduler = define_scheduler(optimizer, config)
    """
        @ justify the resume model
    """
    if config['resume'] != 'None':
        checkpoint = torch.load(config['resume'],
                                map_location=torch.device('cpu'))
        resume = checkpoint['epoch']
        lowest_loss = checkpoint['loss']
        best_epoch = checkpoint['best_epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O0',
                                          verbosity=0)
        amp.load_state_dict(checkpoint['amp'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            resume, checkpoint['epoch']))
        del checkpoint
    log_file = 'log_train_start_{}.txt'.format(resume)
    """
        @ convert model to multi-gpus modes for training
    """
    model = apex.parallel.convert_syncbn_model(model)
    if config['resume'] == 'None':
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O0',
                                          verbosity=0)
    model = DDP(model)
    if gpu == 0:
        sys.stdout = Logger(os.path.join(save_folder, log_file))
    print("Number of parameters in model is {:.3f}M".format(
        sum(tensor.numel() for tensor in model.parameters()) / 1e6))
    """
        @ start to train
    """
    for epoch in range(resume + 1, config['epoches'] + 1):
        print('=> Starch Epoch {}\n'.format(epoch))
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('learning rate is set to {}.\n'.format(
            optimizer.param_groups[0]['lr']))
        model.train()
        sampler_train.set_epoch(epoch)
        batch_time = AverageMeter()
        losses = AverageMeter()
        metric_train = Metrics()
        rmse_train = AverageMeter()
        mae_train = AverageMeter()
        time_snap = time.time()
        for i, inputs in tqdm(enumerate(loader_train)):
            gt, noise = inputs['gt'].cuda(gpu), inputs['noise'].cuda(gpu)
            optimizer.zero_grad()
            """ update the train inputs
            """
            # patten = np.random.randint(0, 4, 1)
            patten = torch.randint(0, 8, (1, ))
            redinput, blueinput = mask_sampling(noise, patten)

            # redinput, blueinput = generator(noise, mask1, mask2)
            output = model(redinput)
            loss = criterion_1(output, blueinput)
            fulloutput = model(noise)
            redoutput, blueoutput = mask_sampling(fulloutput, patten)
            # redoutput, blueoutput = generator(fulloutput, mask1, mask2)

            loss2 = criterion_2(output, blueinput, redoutput, blueoutput)
            losssum = config["gamma"] * loss2 + loss
            with amp.scale_loss(losssum, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            "@ map-reduce tensor"
            rt = reduce_tensor(losssum.data)
            torch.cuda.synchronize()
            losses.update(rt.item(), loader_train.batch_size)
            metric_train.calculate(fulloutput.detach(), gt)
            rmse_train.update(metric_train.get_metric('mse'), metric_train.num)
            mae_train.update(metric_train.get_metric('mae'), metric_train.num)
            batch_time.update(time.time() - time_snap)
            time_snap = time.time()
            if (i + 1) % config['print_freq'] == 0:
                if gpu == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                          'Metric {rmse_train.val:.6f} ({rmse_train.avg:.6f})'.
                          format(epoch,
                                 i + 1,
                                 len(loader_train),
                                 batch_time=batch_time,
                                 loss=losses,
                                 rmse_train=rmse_train))
            if (i + 1) % config['save_freq'] == 0:
                print('=> Start sub-selection validation set')
                rmse, mae = val(model, loader_val, epoch, gpu)
                model.train()
                if gpu == 0:
                    print("===> Average RMSE score on selection set is {:.6f}".
                          format(rmse))
                    print("===> Average MAE score on selection set is {:.6f}".
                          format(mae))
                    print(
                        "===> Last best score was RMSE of {:.6f} in epoch {}".
                        format(lowest_loss, best_epoch))

                    if rmse > lowest_loss:
                        lowest_loss = rmse
                        best_epoch = epoch
                        states = {
                            'epoch': epoch,
                            'best_epoch': best_epoch,
                            'loss': lowest_loss,
                            'state_dict': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'amp': amp.state_dict()
                        }

                        save_checkpoints(states, save_folder, epoch, gpu, True)
        # save checkpoints
        print('=> Start selection validation set')
        rmse, mae = val(model, loader_val, epoch, gpu)
        model.train()
        if gpu == 0:
            print("===> Average RMSE score on selection set is {:.6f}".format(
                rmse))
            print("===> Average MAE score on selection set is {:.6f}".format(
                mae))
            print("===> Last best score was RMSE of {:.6f} in epoch {}".format(
                lowest_loss, best_epoch))
            if rmse > lowest_loss:
                best_epoch = epoch
                lowest_loss = rmse
                states = {
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'loss': lowest_loss,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict()
                }
                save_checkpoints(states, save_folder, epoch, gpu, True)

        if config['lr_policy'] == 'plateau':
            scheduler.step(rmse)
        else:
            scheduler.step()
        # if (epoch) % 10 == 0:
        #     config["gamma"] += 0.5
        print('=>> the model training finish!')


def save_checkpoints(model_state, save_folder, epoch, gpuid, is_best=False):
    # filepath = os.path.join(save_folder, 'checkpoint_{}.pt'.format(gpuid))
    # print('save the current model : {} \n'.format(filepath))
    if is_best:
        torch.save(model_state,
                   os.path.join(save_folder, 'model_best_{}.pt'.format(gpuid)))
        print('Best model in epoch {} copied!! \n'.format(epoch))


def test(config):
    # prepapre the dataset
    dataset = get_data(config)
    testset = dataset(config, "test")
    loader_test = DataLoader(dataset=testset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=config['num_threads'])
    model = UNet(config["in_channels"], config["out_channels"])
    if config['resume'] != 'None':
        checkpoint = torch.load(config['resume'],
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print("Load pretrained model finish!")
        del checkpoint
    model.cuda()

    psnrs = []
    ssims = []
    model.eval()
    with torch.no_grad():
        for _, inputs in tqdm(enumerate(loader_test)):
            noise = inputs['noise'].cuda()
            output = model(noise)
            gt = inputs['gt'].squeeze().numpy()
            output = output.squeeze().cpu().numpy()
            psnr = compare_psnr(gt, output)
            ssim = compare_ssim(gt, output)
            psnrs.append(psnr)
            ssims.append(ssim)
    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    print("PSNR, SSIM:{},{}\n".format(psnrs.mean(), ssims.mean()))


def main(config):
    if not config['eval']:
        if config['num_gpus'] <= 1:
            train(0, config)
        else:
            spawn_context = mp.spawn(train,
                                     nprocs=config['num_gpus'],
                                     args=(config, ),
                                     join=False)
        while not spawn_context.join():
            pass
        for process in spawn_context.processes:
            if process.is_alive():
                process.terminate()
            process.join()

    test(config)


if __name__ == "__main__":
    main(config)