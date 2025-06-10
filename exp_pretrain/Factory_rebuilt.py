import argparse
import os
import random
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 对数据进行标准化处理和归一化处理的包
from pathlib import Path

from utils.Memory import Memory
from utils.Timefeatures import time_features
from utils.Layer_Tools import *

data_columns = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM25', 'wd', 'ws', 'T',
                'P', 'RH', 'Bare', 'Building', 'Forest', 'Grass', 'OISA', 'Road',
                'Water', 'elevation', 'slope', 'aspect', 'hillshade', 'road_density',
                'building_height', 'D2S', 'poi_交通设施', 'poi_休闲娱乐', 'poi_公司企业',
                'poi_医疗健康', 'poi_商务住宅', 'poi_旅游景点', 'poi_汽车相关', 'poi_生活服务', 'poi_科教文化',
                'poi_购物消费', 'poi_运动健身', 'poi_酒店住宿', 'poi_金融机构', 'poi_餐饮美食']

columns_dict = {column: index for index, column in enumerate(data_columns)}


# 将csv文件或parquet文件转化为pt文件
def convert_table_to_pt_per_site(
        data_path: Path,
        data_suffix: str,
        save_dir: str,
        target_columns: list,
        date_column='date',
        site_column='站点',
        sites=None,
        freq='h',
        replace_value=0
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if data_suffix == '.csv':
        print(f'读取CSV文件:{data_path}')
        df_raw = pd.read_csv(data_path, encoding='utf-8')
    elif data_suffix == '.parquet':
        print(f'读取Parquet文件:{data_path}')
        df_raw = pd.read_parquet(data_path)
    else:
        raise ValueError()

    df_raw.replace([-9999, np.nan], replace_value, inplace=True)
    index_map = {}

    for site in sites:
        df_site = df_raw[df_raw[site_column] == site]
        # 保存目标变量数据
        data = df_site[target_columns].to_numpy(dtype=np.float32)
        data_tensor = torch.tensor(data)  # shape: [T, D]
        data_path = save_dir / f"{site}_data.pt"
        torch.save(data_tensor, data_path)

        # 保存时间特征
        date_series = pd.to_datetime(df_site[date_column].values)
        time_feat = time_features(date_series, freq=freq)  # shape assumed to be [1, D, T, 1] or similar
        time_feat_tensor = torch.tensor(time_feat, dtype=torch.float32).permute(1, 0)
        stamp_path = save_dir / f"{site}_stamp.pt"
        torch.save(time_feat_tensor, stamp_path)

        # 索引信息更新
        index_map[site] = {
            "data_path": str(data_path),
            "stamp_path": str(stamp_path)
        }

    # 保存站点索引映射
    with open(save_dir / "sites.json", "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)

    print("转换完成！共保存：")
    print(f"- {len(sites)} 个站点 .pt 文件")
    print(f"- 索引文件：{save_dir / 'sites.json'}")


def to_pt(data_set):  # 初次读取时，将csv文件或parquet转化为pt文件方便读取
    file_suffix = Path(data_set.data_path).suffix
    default_pt_dir = rf'..\data\pts\{data_set.mission}'

    if file_suffix == '.csv':
        pass
    elif file_suffix == '.parquet':
        pass
    else:
        raise ValueError()

    convert_table_to_pt_per_site(
        data_path=data_set.data_path,
        data_suffix=file_suffix,
        save_dir=default_pt_dir,
        target_columns=data_columns,
        sites=data_set.sites
    )


class Pretrain_Dataset(Dataset):
    def __init__(self, data_path: Optional[Path], memory: Optional[Memory], sites: list, scale: Optional[str], freq: str,
                 size: Optional[list], target: list, context_window: int, target_window: int, mission: str,
                 add_influence: bool):
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
        self.mission = mission
        self.columns_dict = columns_dict
        self.add_influence = add_influence

        self.__read_data__()

        # 针对target，在memory中读取相应的列
        if self.memory is not None:
            self.influence_index_list = []
            for pollutant in self.target:
                influence_name_list = self.memory.read(pollutant)
                influence_index = [self.columns_dict[influence_name] for influence_name in influence_name_list]
                self.influence_index_list.append(influence_index)

            self.target_index = [self.columns_dict[target] for target in self.target]
        else:
            self.influence_index_list=None
            self.target_index = None

    def __read_data__(self):
        sites_json_path = rf'..\data\pts\{self.mission}\sites.json'
        if not os.path.exists(sites_json_path):
            if self.data_path is None:
                pass
            elif Path(self.data_path).suffix in ['.csv', '.parquet']:
                to_pt(self)

        # 每个.pt文件保存了形状为[8736, 39]的张量
        with open(sites_json_path, 'r', encoding='utf-8') as f:
            self.index_map = json.load(f)

        self.sites = list(self.index_map.keys())

        self.data = {}
        self.stamps = {}
        self.lengths = []

        for site in self.sites:
            paths = self.index_map[site]
            data_tensor = torch.load(paths["data_path"], weights_only=True)  # [T, D]
            stamp_tensor = torch.load(paths["stamp_path"], weights_only=True)

            self.data[site] = data_tensor
            self.stamps[site] = stamp_tensor

            valid_len = data_tensor.shape[0] - self.context_window - self.target_window + 1
            self.lengths.append(valid_len)

        # 计算长度数组的累加和
        self.cumulative_lengths = np.cumsum(self.lengths)

    def __getitem__(self, index):
        site_index = np.searchsorted(self.cumulative_lengths, index, side='right')
        # local_index：表示在站点数据中的索引
        if site_index > 0:
            local_index = index - self.cumulative_lengths[site_index - 1]
        else:
            local_index = index
        site = self.sites[site_index]

        # 获取站点的全部数据
        site_data = self.data[site]  # [8736, 39]
        site_stamp = self.stamps[site]  # [8736, 4]

        x = site_data[:, self.target_index][local_index : local_index+self.context_window]
        y = site_data[:, self.target_index][local_index+self.context_window : local_index+self.context_window+self.target_window]
        stamp = site_stamp[local_index : local_index+self.context_window]

        got_data = {
            'x': x.permute(1, 0),
            'y': y.permute(1, 0),
            'stamp': stamp
        }

        if self.add_influence is None:
            return got_data
        else:
            full_data = site_data[local_index : local_index+self.context_window]
            got_data['full_data'] = full_data.permute(1, 0)
            return got_data

    def __len__(self):
        return sum(self.lengths)

    # 获取不同污染物的均值和方差
    @property
    def statistic(self):
        datas = pd.read_parquet(self.data_path)
        mean_list = []
        std_list = []
        for item in self.target:
            mean_list.append(datas[item].mean())
            std_list.append(datas[item].std())

        statistic_dict = {
            'means': mean_list,
            'stds': std_list
        }
        return statistic_dict


def pretrained_dataset_provider(args):
    memory = None if args.memory is None else Memory(args.memory)
    if args.mission == ' test':
        file_path = Path(rf'..\{args.mission}_sites.txt')
        sites = file_path.read_text(encoding='utf-8').splitlines()
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

        data_set = Pretrain_Dataset(
            args.data_path,
            memory,
            sites,
            args.scale,
            args.freq,
            args.size,
            args.target,
            args.context_window,
            args.target_window,
            args.mission,
            args.add_influence
        )

    elif args.mission == 'vali':
        file_path = Path(rf'..\{args.mission}_sites.txt')
        sites = file_path.read_text(encoding='utf-8').splitlines()
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

        data_set = Pretrain_Dataset(
            args.data_path,
            memory,
            sites,
            args.scale,
            args.freq,
            args.size,
            args.target,
            args.context_window,
            args.target_window,
            args.mission,
            args.add_influence
        )

    else:
        file_path = Path(rf'..\{args.mission}_sites.txt')
        sites = file_path.read_text(encoding='utf-8').splitlines()
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

        data_set = Pretrain_Dataset(
            args.data_path,
            memory,
            sites,
            args.scale,
            args.freq,
            args.size,
            args.target,
            args.context_window,
            args.target_window,
            args.mission,
            args.add_influence
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,  # 用于加载的子进程数量
        drop_last=drop_last,  # 是否舍弃最后一个不足一个batch_size的批次
        pin_memory=True  # 是否将加载的数据张量固定在内存中
    )

    return data_set, data_loader


if __name__ == '__main__':
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
    parser.add_argument('--data_path', type=str,
                        default=rf'..\data\samples\merged_air_meteo_data_lc_newtestsite_train_stride0.parquet')
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
    parser.add_argument('--memory', type=str, default=r'..\Memories.txt')
    parser.add_argument('--add_influence', type=bool, default=True)

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
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--fine-tuned', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--loss_individual', type=bool, default=False)
    parser.add_argument('--pretrained_checkpoints', type=str, default=rf'D:\gzr\project_sz\Pretrained\checkpoints')

    args = parser.parse_args()

    data_set, data_loader = pretrained_dataset_provider(args)

    print(data_set.statistic)
