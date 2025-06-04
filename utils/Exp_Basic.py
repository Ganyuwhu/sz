import argparse
import os
import torch
import json
from datetime import date


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        target_str = '-'.join(args.target)
        today = date.today()
        date_str = today.strftime("%Y-%m-%d")
        self.setting = '{}_{}_{}_pl{}_fl{}_dm{}_dff{}_{}'.format(
            target_str,
            args.model,
            args.scale if args.scale is not None else 'No_type',
            args.context_window,
            args.target_window,
            args.d_model,
            args.d_ff,
            date_str
        )

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu) if not self.args.use_multi_gpu \
                else self.args.devices
            # os.environ是一个环境变量，'CUDA_VISIBLE_DEVICES'用来声明哪些gpu可以被访问
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('使用GPU:cuda:{}'.format(self.args.gpu))

        else:
            device = torch.device('cpu')
            print('使用CPU，运行缓慢')

        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


# 定义类型转换函数
def parse_dict(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")
