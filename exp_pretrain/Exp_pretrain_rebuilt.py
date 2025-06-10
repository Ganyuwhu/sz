import warnings

from tqdm import tqdm

from utils.Exp_Basic import Exp_Basic
from Factory_rebuilt import pretrained_dataset_provider
from Models import patchTST, patchTST2, patchTST_multiChannel
from utils.Learning_Tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.Metrics import *
from utils.Memory import Memory
from collections import defaultdict

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import argparse
import random

warnings.filterwarnings("ignore", category=RuntimeWarning)


def criterion():
    return nn.MSELoss()


def calculate_average_values(dict_list):
    # 创建一个默认值为列表的字典来存储每个键的所有值
    key_values = defaultdict(list)

    # 遍历列表中的每个字典
    for d in dict_list:
        # 遍历字典中的每个键值对
        for key, value in d.items():
            key_values[key].append(value)

    # 计算每个键的平均值
    averages = {}
    for key, values in key_values.items():
        averages[key] = sum(values) / len(values)

    return averages


class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.data_set, self.data_loader = pretrained_dataset_provider(self.args)
        self.criterion = criterion()
        self._get_data()
        self.args.influence_index_list = self.data_set[f'{self.args.mission}_dataset'].influence_index_list
        self.model = self._build_model().to(self.device)


    def _build_model(self):
        model_dict = {
            'patchTST': patchTST,
            'patchTST2': patchTST2,
            'patchTST_multiChannel': patchTST_multiChannel
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model = model.to(self.args.device_ids[0])  # 主设备设为 cuda:0
        else:
            model = model.to(self.device)

        return model

    def _get_data(self):
        self.data_set = {}
        self.data_loader = {}
        if self.args.mission == 'test':
            self.data_set['test_dataset'], self.data_loader['test_dataloader'] = pretrained_dataset_provider(self.args)
        elif self.args.mission == 'train':
            self.data_set['train_dataset'], self.data_loader['train_dataloader'] = pretrained_dataset_provider(self.args)
            self.args.mission = 'vali'
            self.data_set['vali_dataset'], self.data_loader['vali_dataloader'] = pretrained_dataset_provider(self.args)
            self.args.mission = 'train'

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, **kwargs):
        vali_loss = []
        metrics_list = []
        vali_metrics = metrics()
        if 'load_model' in kwargs and kwargs['load_model'] is True:
            if 'model_path' in kwargs:
                try:
                    self.model = torch.load(kwargs['model_path']).to(self.device)
                except (FileNotFoundError, RuntimeError) as e:
                    print(f"Failed to load model: {e}")

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader['vali_dataloader']):
                batch_x, batch_y, batch_stamp = batch['x'].to(self.device), batch['y'].to(self.device), batch[
                    'stamp'].to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.add_influence:
                            batch_full_data = batch['full_data'].to(self.device)
                            outputs = self.model(batch_x, batch_stamp, batch_full_data)
                        else:
                            outputs = self.model(batch_x, batch_stamp)

                else:
                    if self.args.add_influence:
                        batch_full_data = batch['full_data'].to(self.device)
                        outputs = self.model(batch_x, batch_stamp, batch_full_data)
                    else:
                        outputs = self.model(batch_x, batch_stamp)

                predict = outputs['predict']

                loss = torch.sqrt(self.criterion(predict, batch_y))

                vali_loss.append(loss.item())
                metrics_list.append(vali_metrics(predict, batch_y))

            total_loss = np.average(vali_loss)
            vali_metrics = calculate_average_values(metrics_list)

            self.model.train()
            return total_loss, vali_metrics

    def train(self):
        path = os.path.join(self.args.pretrained_checkpoints, self.setting)
        train_loader = self.data_loader['train_dataloader']
        train_dataset = self.data_set['train_dataset']

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params}')

        train_epochs = self.args.train_epochs
        print(f'epochs:{train_epochs}')

        # 计算batch_y的标准差和均值作为模型先验
        statistic_dict = train_dataset.statistic
        total_mean = torch.tensor(statistic_dict['means'])
        total_std = torch.tensor(statistic_dict['stds'])
        with torch.no_grad():  # 避免不必要的梯度计算
            self.model.model.beta.data.copy_(total_mean.to(self.device))
            self.model.model.gamma.data.copy_(total_std.to(self.device))
        print(self.model.model.beta, self.model.model.gamma)

        epoch_train_loss = []
        epoch_vali_loss = []

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            metrics_list = []
            train_metrics = metrics()

            self.model.train()
            epoch_time = time.time()

            for i, batch in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_stamp = batch['x'].to(self.device), batch['y'].to(self.device), batch[
                    'stamp'].to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.add_influence:
                            batch_full_data = batch['full_data'].to(self.device)
                            outputs = self.model(batch_x, batch_stamp, batch_full_data)
                        else:
                            outputs = self.model(batch_x, batch_stamp)

                else:
                    if self.args.add_influence:
                        batch_full_data = batch['full_data'].to(self.device)
                        outputs = self.model(batch_x, batch_stamp, batch_full_data)
                    else:
                        outputs = self.model(batch_x, batch_stamp)

                predict = outputs['predict']

                if torch.isnan(predict).any():
                    print('output error')
                    if torch.isnan(batch_x).any():
                        print('batch_x error')
                    if torch.isnan(batch_stamp).any():
                        print('batch_stamp error')

                loss = torch.sqrt(self.criterion(predict, batch_y))

                train_loss.append(loss.item())
                epoch_metrics = train_metrics(predict, batch_y)
                metrics_list.append(epoch_metrics)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # for key in epoch_metrics:
                    #     print(key, ': ', epoch_metrics[key])
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.array(train_loss)
            max_val = np.nanmax(train_loss)  # 忽略NaN计算最大值
            train_loss[np.isnan(train_loss)] = max_val
            train_loss = np.average(train_loss)
            vali_loss, vali_metrics = np.array(self.vali())

            epoch_train_loss.append(train_loss)
            epoch_vali_loss.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            train_metrics = calculate_average_values(metrics_list)
            print('train_metrics:')
            for key in train_metrics:
                print(key, ': ', train_metrics[key])
            print('vali_metrics:')
            for key in vali_metrics:
                print(key, ': ', vali_metrics[key])

            torch.save(self.model, path + '/' + 'checkpoint_' + str(epoch) + '.pth')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(self.model, best_model_path)

        np.save(rf'..\exp_pretrain\result\{self.setting}_train_loss.npy', epoch_train_loss)
        np.save(rf'..\exp_pretrain\result\{self.setting}_vali_loss.npy', epoch_vali_loss)

        return self.model

    def test(self, **kwargs):
        test_loader = self.data_loader
        model_path = os.path.join(self.args.pretrained_checkpoints, self.setting, 'checkpoint.pth')
        result_path = os.path.join(self.args.pretrained_checkpoints, self.setting)

        # metrics
        test_loss = []
        metrics_list = []
        test_metrics = metrics()

        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        try:
            self.model = torch.load(model_path).to(self.device)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Failed to load model: {e}")

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader['test_dataloader']):
                batch_x = batch['x'].to(self.device)
                batch_y = batch['y'].to(self.device)
                batch_stamp = batch['stamp'].to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.memory:
                            batch_full_data = batch['full_data'].to(self.device)
                            outputs = self.model(batch_x, batch_stamp, batch_full_data)
                        else:
                            outputs = self.model(batch_x, batch_stamp)

                else:
                    if self.args.memory:
                        batch_full_data = batch['full_data'].to(self.device)
                        outputs = self.model(batch_x, batch_stamp, batch_full_data)
                    else:
                        outputs = self.model(batch_x, batch_stamp)

                predict = outputs['predict']

                loss = torch.sqrt(self.criterion(predict, batch_y))

                test_loss.append(loss.item())
                metrics_list.append(test_metrics(predict, batch_y))

            test_loss = np.average(test_loss)
            test_metrics = calculate_average_values(metrics_list)

            print(f'test_loss: {test_loss}')
            print(f'test_metrics: {test_metrics}')

