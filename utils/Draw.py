import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def draw(npy_files:list):
    array_list = []
    name_list = []
    for npy_file in npy_files:
        name_list.append(Path(npy_file).stem)
        array_list.append(np.load(npy_file))

    plt.figure(figsize=(10, 6))
    for name, array in zip(name_list, array_list):
        plt.plot(array, label=name)
    plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    plt.legend()
    plt.show()


if __name__ == '__main__':
    npy_files = [
        r'..\exp_pretrain\result\patchTST_train_loss.npy',
        r'..\exp_pretrain\result\patchTST_vali_loss.npy',
        r'..\exp_pretrain\result\patchTST2_train_loss.npy',
        r'..\exp_pretrain\result\patchTST2_vali_loss.npy',
    ]

    draw(npy_files)
