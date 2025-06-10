import argparse
import random

import numpy as np
import torch

from Exp_pretrain_rebuilt import Exp_Pretrain


def pretrain_patchTST2():
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')  # required=True,
    parser.add_argument('--model', type=str, default='patchTST2', help='model name')

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
    parser.add_argument('--add_influence', type=bool, default=True)
    parser.add_argument('--memory', type=str, default=r'..\Memories.txt')
    parser.add_argument('--memory_list', type=list, default=None)

    # optim
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--train_epochs', type=int, default=50)

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
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--loss_individual', type=bool, default=False)
    parser.add_argument('--pretrained_checkpoints', type=str, default=rf'..\Pretrained\checkpoints')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    exp_main = Exp_Pretrain

    exp = exp_main(args)
    if args.mission == 'train':
        print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(exp.setting))
        exp.train()
    else:
        print('>>>>>>>pretrained model testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(exp.setting))
        exp.test()


if __name__ == '__main__':
    pretrain_patchTST2()
