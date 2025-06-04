import argparse
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 对数据进行标准化处理和归一化处理的包
from pathlib import Path

from utils.Memory import Memory
from utils.Timefeatures import time_features
from utils.Layer_Tools import *


class Pretrain_Dataset(Dataset):
    def __init__(self, data_path: Path, memory: Optional[Memory], sites: list, scale: Optional[str], freq: str,
                 size: Optional[list], target: list, context_window: int, target_window: int):
        super().__init__()

        self.data_path = data_path
        self.memory = memory
        self.sites = sites
        self.scale = scale
        self.freq = freq
        self.size = size
        self.target = target
        self.context_window = context_window
        self.target_window = target_window

        self.__read_data__()

    def __read_data__(self):
        print("-------------------read pretrained dataset data-------------------")
        # 初始化归一化
        if self.scale:
            if self.scale == 'Standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()

        df_raw = pd.read_csv(self.data_path, encoding='utf-8')
        df_raw.replace([-9999, np.nan], 0, inplace=True)

        # 对DataFrame的列进行标号
        cols_to_delete = ['date', '站点']
        columns = list(set(df_raw.columns) - set(cols_to_delete))

        self.columns_dict = {col_name: idx for idx, col_name in enumerate(columns)}

        # 分站点读取数据
        self.data_dict = {site: df_raw[df_raw['站点'] == site].drop(columns=cols_to_delete) for site in self.sites}

        # 清理target含零量过高的站点
        keys_to_delete = [
            k for k, v in self.data_dict.items()
            if np.sum(v[self.target].to_numpy() == 0).sum() / len(v) >= 0.3
        ]
        print(f'{len(keys_to_delete)}个含零量过高被剔除的站点：{keys_to_delete}')
        for k in keys_to_delete:
            del self.data_dict[k]

        # 更新域
        self.sites = list(self.data_dict.keys())

        # 处理零值
        for site in self.sites:
            self.data_dict[site][self.target] = fix_zeros_optimized(self.data_dict[site][self.target])

        # 处理时间戳
        self.data_stamp = {site: time_features(pd.to_datetime(df_raw[df_raw['站点'] == site]['date'].values),
                                               freq=self.freq).transpose(1, 0) for site in self.sites}

        # 获取各站点可用序列数
        self.data_len = [len(self.data_dict[site]) - self.context_window - self.target_window + 1 for site in
                         self.data_dict.keys()]

    def __getitem__(self, index):
        site_index = 0
        for data_len in self.data_len:
            if index < data_len:
                break
            else:
                index = index - data_len
                site_index = site_index + 1

        data = self.data_dict[self.sites[site_index]][self.target].to_numpy()
        data_x = data[index:index + self.context_window]
        data_y = data[index + self.context_window:index + self.context_window + self.target_window]
        data_stamp = self.data_stamp[self.sites[site_index]][index: index + self.context_window]

        data_x = torch.tensor(data_x, dtype=torch.float32).permute(-1, -2)
        data_y = torch.tensor(data_y, dtype=torch.float32).permute(-1, -2)
        data_stamp = torch.tensor(data_stamp, dtype=torch.float32)

        if self.memory:
            data_full = self.data_dict[self.sites[site_index]].to_numpy()[index:index + self.context_window]
            data_full = torch.tensor(data_full, dtype=torch.float32).permute(-1, -2)
            return data_x, data_y, data_stamp, data_full
            pass
        else:
            return data_x, data_y, data_stamp

    def __len__(self):
        return np.sum(self.data_len)

    @property
    def Memory(self):
        memory_list = []
        if self.memory is None:
            return None
        for pollutant in self.target:
            influence = self.memory.read(pollutant)
            memory_list.append([self.columns_dict[cols] for cols in influence])

        return memory_list


def pretrained_dataset_provider(args):
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    memory = None if args.memory is None else Memory(args.memory)
    if args.mission == 'test':
        file_path = parent_dir / 'test_sites.txt'
        data_path = parent_dir.parent / 'sz' / 'samples' / 'merged_air_meteo_data_lc_newtestsite_test_stride0.csv'
        sites = file_path.read_text(encoding='utf-8').splitlines()
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size

        data_set = Pretrain_Dataset(
            data_path,
            memory,
            sites,
            args.scale,
            args.freq,
            args.size,
            args.target,
            args.context_window,
            args.target_window,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,  # 用于加载的子进程数量
            drop_last=drop_last,  # 是否舍弃最后一个不足一个batch_size的批次
            pin_memory=True  # 是否将加载的数据张量固定在内存中
        )

    else:
        file_path_train = parent_dir / 'train_sites.txt'
        file_path_vali = parent_dir / 'vali_sites.txt'
        data_path = parent_dir.parent / 'sz' / 'samples' / 'merged_air_meteo_data_lc_newtestsite_train_stride0.csv'
        train_sites = file_path_train.read_text(encoding='utf-8').splitlines()
        vali_sites = file_path_vali.read_text(encoding='utf-8').splitlines()
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size

        data_set = []
        data_loader = []

        train_set = Pretrain_Dataset(
            data_path,
            memory,
            train_sites,
            args.scale,
            args.freq,
            args.size,
            args.target,
            args.context_window,
            args.target_window,
        )
        data_set.append(train_set)

        val_set = Pretrain_Dataset(
            data_path,
            memory,
            vali_sites,
            args.scale,
            args.freq,
            args.size,
            args.target,
            args.context_window,
            args.target_window,
        )
        data_set.append(val_set)

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,  # 用于加载的子进程数量
            drop_last=drop_last,  # 是否舍弃最后一个不足一个batch_size的批次
            pin_memory=True  # 是否将加载的数据张量固定在内存中
        )
        data_loader.append(train_loader)

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,  # 用于加载的子进程数量
            drop_last=drop_last,  # 是否舍弃最后一个不足一个batch_size的批次
            pin_memory=True  # 是否将加载的数据张量固定在内存中
        )
        data_loader.append(val_loader)

    return data_set, data_loader


if __name__ == "__main__":
    # 简单测试一下怎么使用memory
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')  # required=True,
    parser.add_argument('--model', type=str, default='patchTST', help='model name')

    # Prediction
    parser.add_argument('--context_window', type=int, default=336, help='input sequence length')
    parser.add_argument('--target_window', type=int, default=168, help='prediction sequence length')

    # dataloader
    # parser.add_argument('--data_path', type=str,
    #                     default=rf"D:\gzr\sz\samples\merged_air_meteo_data_pretrain_withtest.csv")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--scaler_path', type=str, default=None)
    parser.add_argument('--scale', type=str, default=None)
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--size', type=list, default=[336, 168])
    parser.add_argument('--target', type=list, default=['NO2', 'O3', 'PM25'])

    # model
    parser.add_argument('--patch_len', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--pe', type=str, default='sincos')
    parser.add_argument('--learn_pe', type=bool, default=False)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=None)
    parser.add_argument('--d_v', type=int, default=None)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--norm', type=str, default='LayerNorm')
    parser.add_argument('--attn_dropout', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--res_attention', type=bool, default=True)
    parser.add_argument('--pre_norm', type=bool, default=False)
    parser.add_argument('--pretrain_head', type=bool, default=False)
    parser.add_argument('--head_type', type=str, default='flatten')
    parser.add_argument('--memory', type=str, default=None)

    # optim
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--train_epochs', type=int, default=100)

    # device
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    parser.add_argument('--num_workers', type=int, default=20)

    # exp
    parser.add_argument('--mission', type=str, default='test')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--fine-tuned', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--loss_individual', type=bool, default=False)
    parser.add_argument('--pretrained_checkpoints', type=str, default=rf'D:\gzr\project_sz\Pretrained\checkpoints')

    args = parser.parse_args()

    data_set, data_loader = pretrained_dataset_provider(args)
    data_example = data_set[0]
    print(data_example[0].shape, data_example[1].shape, data_example[2].shape, data_example[3].shape)
    print(data_set.Memory)
