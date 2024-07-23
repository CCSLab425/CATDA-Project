"""
Author: Yin Hua(CCS Lab)
"""

import numpy as np
import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch import nn
from models.SCARA_res_net import SCARA_ViT_model
import os
import argparse
from dataloader import dataload, test_dataload
from utils_dsbn.save_other_functions import DDC_train_history, save_his
from MMD.MMD_calculation import MMDLoss
from data import build_loader
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--num_class', type=int, default=10)    #10or4
parser.add_argument('--gamma1', type=float, default=2.5)
parser.add_argument('--gamma2', type=float, default=2)
parser.add_argument('--lr', type=float, default=1e-3)   
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--diagnostic_dataset', type=str, default='bearing')
parser.add_argument('--dateset_path', type=str, default='./dataset/bearing_datasets/') 
parser.add_argument('--src_dateset_name', type=str, default='SUST_500.mat')     
parser.add_argument('--tgt_dateset_name', type=str, default='SUST_2000.mat')
parser.add_argument('--model_path', type=str, default='save_dir/')
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
train_hist = DDC_train_history()

def test(model, dataset_name_src,dataset_name_tar):
    loss_fn = torch.nn.CrossEntropyLoss()
    loader_src_test = test_dataload(batch_size=args.batch_size, dataset_path=dataset_name_src)
    loader_tar_test = test_dataload(batch_size=args.batch_size, dataset_path=dataset_name_tar)
    model.eval()
    with torch.no_grad():
        n_samples_s = 0
        n_samples_t = 0
        n_correct_s = 0
        n_correct_t = 0
        loss_per_epoch_s = []
        loss_per_epoch_t = []
        for (data_src, data_tar) in zip(enumerate(loader_src_test), enumerate(loader_tar_test)):
            _, (x_src, y_src) = data_src
            _, (x_tar, y_tar) = data_tar
            n_samples_s += len(y_src)
            n_samples_t += len(y_tar)
            x_src, y_src, x_tar ,y_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE) ,y_tar.to(DEVICE)
            x_s_t=None

            ys_pre, ys_t_pre, yt_pre = model(x_src, x_s_t , x_tar)

            pred_s = torch.argmax(ys_pre, 1)
            loss_s = loss_fn(ys_pre, pred_s)
            n_correct_s += (pred_s == y_src.long()).sum().item()
            loss_per_epoch_s.append(loss_s.item())
            pred_t = torch.argmax(yt_pre, 1)
            loss_t = loss_fn(yt_pre, pred_t)
            n_correct_t += (pred_t == y_tar.long()).sum().item()
            loss_per_epoch_t.append(loss_t.item())
    acc_s = float(n_correct_s) / n_samples_s
    loss_mean_s = np.mean(loss_per_epoch_s)
    acc_t = float(n_correct_t) / n_samples_t
    loss_mean_t = np.mean(loss_per_epoch_t)

    return acc_s, loss_mean_s , acc_t, loss_mean_t,


def train(model, optimizer, dataloader_src, dataloader_tar, save_name=''):
    loss_class = torch.nn.CrossEntropyLoss()
    best_acc = -float('inf')
    src_tgt_mmd_loss_per_epoch = []
    yhmmd_loss_per_epoch = []
    total_loss_per_epoch = []

    for epoch in range(args.n_epoch):
        model.train()
        for (data_src, data_tar) in zip(enumerate(dataloader_src), enumerate(dataloader_tar)):
            _, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
            x_s_t=None

            ys_pre, ys_t_pre, yt_pre,  = model(x_src, x_s_t , x_tar)
            src_loss = loss_class(ys_pre, y_src)
            mmd_loss = MMDLoss()
            DDC_loss1 = mmd_loss(ys_pre, ys_t_pre)
            DDC_loss2= mmd_loss(ys_pre, yt_pre)
            YHMMD_loss = args.gamma1*DDC_loss1 + args.gamma2*DDC_loss2
            total_loss = src_loss + YHMMD_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            src_tgt_mmd_loss_per_epoch.append(DDC_loss1.item())
            yhmmd_loss_per_epoch.append(YHMMD_loss.item())
            total_loss_per_epoch.append(total_loss.item())
        src_tgt_mmd_loss_mean = np.mean(src_tgt_mmd_loss_per_epoch)
        yhmmd_loss_mean = np.mean(yhmmd_loss_per_epoch)
        total_loss_mean = np.mean(total_loss_per_epoch)
        item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, DDC_loss1: {:.4f}, DDC_loss2: {:.4f},YHMMD_loss: {:.4f},total_loss: {:.4f}'.format(
            epoch, args.n_epoch, src_loss.item(), DDC_loss1.item(),DDC_loss2.item(),YHMMD_loss.item() ,total_loss.item())
        print(item_pr, end=' >>> ')
        
        # test
        src_test_acc, src_test_loss ,tgt_test_acc, tgt_test_loss = test(model, args.dateset_path + args.src_dateset_name , args.dateset_path + args.tgt_dateset_name)
        test_info = 'Source acc: {:.3f} %, Target acc: {:.3f} %'.format(src_test_acc * 100, tgt_test_acc * 100)
        print(test_info)

        if best_acc < tgt_test_acc:
            best_acc = tgt_test_acc
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            torch.save(model, '{}.pth'.format(args.model_path + save_name))

        train_hist['src_and_tgt_mmd_loss'].append(src_tgt_mmd_loss_mean)
        train_hist['yhmmd_loss'].append(yhmmd_loss_mean)
        train_hist['total_loss'].append(total_loss_mean)
        train_hist['Source_test_acc'].append(src_test_acc)
        train_hist['Source_test_loss'].append(src_test_loss)
        train_hist['Target_test_acc'].append(tgt_test_acc)
        train_hist['Target_test_loss'].append(tgt_test_loss)


def save_prediction(test_labels):
    prediction = pd.DataFrame(test_labels)
    prediction.to_csv(args.model_path+'test'+'_prediction.csv')


if __name__ == '__main__':
    torch.random.manual_seed(10)
    save_name = args.src_dateset_name[:-4] + '_to_' + args.tgt_dateset_name[:-4]
    print(save_name)
    loader_src = dataload(batch_size=args.batch_size, dataset_path=args.dateset_path + args.src_dateset_name)
    loader_tar = dataload(batch_size=args.batch_size, dataset_path=args.dateset_path + args.tgt_dateset_name)
    model = SCARA_ViT_model(input_size=32, class_num=args.num_class).to(DEVICE)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, loader_src, loader_tar, save_name=save_name)
    save_his(train_hist=train_hist, save_dir=args.model_path, save_name=save_name)
    best_accuracy = np.max(train_hist['Target_test_acc'])
    print('Best test acc: {:.3f} %'.format(best_accuracy * 100))
