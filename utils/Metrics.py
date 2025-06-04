import torch
import numpy as np
import pandas as pd

"""
    predict: [batch_size, num_features, predict_len];
    gt: [batch_size, num_features, predict_len]
"""


class metrics:
    def __init__(self):
        self.metric_dict = {
            'RSE': np.empty(1),
            'RAE': np.empty(1),
            'MSE': np.empty(1),
            'RMSE': np.empty(1),
            'NMSE_1': np.empty(1),
            'NMSE_2': np.empty(1),
            'CORR': np.empty(1),
            'TOP-10': np.empty(1),
            'TOP-30': np.empty(1),
            'Average_Error': np.empty(1)
        }

    def __call__(self, predict, gt):
        predict = predict.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        self.metric_dict['RSE'] = get_rse(predict, gt)
        self.metric_dict['RAE'] = get_rae(predict, gt)
        self.metric_dict['MSE'] = get_mse(predict, gt)
        self.metric_dict['RMSE'] = get_rmse(predict, gt)
        self.metric_dict['NMSE_1'] = get_nmse_1(predict, gt)
        self.metric_dict['NMSE_2'] = get_nmse_2(predict, gt)
        self.metric_dict['CORR'] = get_corr(predict, gt)
        self.metric_dict['TOP-10'] = get_tops(predict, gt, 1-1e-12)
        self.metric_dict['TOP-30'] = get_tops(predict, gt, 0.3)
        self.metric_dict['Average_Error'] = get_avg_error(predict, gt)

        # for key in self.metric_dict:
        #     print(key, ': ', self.metric_dict[key])

        return self.metric_dict


def get_rse(predict, gt):
    upper = np.mean((predict - gt) ** 2, axis=-1)
    lower = np.mean((gt - gt.mean(-1, keepdims=True)) ** 2, axis=-1)
    batch_rse = upper / (lower + 1e-5)
    feature_rse = batch_rse.mean(0)
    return feature_rse


def get_rae(predict, gt):
    predict_flatten = predict.reshape(-1, predict.shape[1])  # [batch_size*predict_len, num_features]
    gt_flatten = gt.reshape(-1, gt.shape[1])

    feature_rae = np.mean(np.abs(predict_flatten - gt_flatten), axis=0)
    return feature_rae


def get_mse(predict, gt):
    predict_flatten = predict.reshape(-1, predict.shape[1])  # [batch_size*predict_len, num_features]
    gt_flatten = gt.reshape(-1, gt.shape[1])

    feature_mse = np.mean((predict_flatten - gt_flatten) ** 2, axis=0)
    return feature_mse


def get_rmse(predict, gt):
    mse = get_mse(predict, gt)
    return np.sqrt(mse)


def get_nmse_1(predict, gt):
    gt_flatten = gt.reshape(-1, gt.shape[1])  # [batch_size*predict_len, num_features]
    rmse = get_rmse(predict, gt)
    gt_mean = np.mean(gt_flatten, axis=0)
    return rmse / (gt_mean + 1e-5)


def get_nmse_2(predict, gt):
    gt_flatten = gt.reshape(-1, gt.shape[1])  # [batch_size*predict_len, num_features]
    rmse = get_rmse(predict, gt)
    gt_std = np.std(gt_flatten, axis=0)
    return rmse / (gt_std + 1e-5)


def get_corr(predict, gt):
    predict = predict.reshape(-1, predict.shape[1])  # [batch_size*predict_len, num_features]
    gt = gt.reshape(-1, gt.shape[1])

    df_predict = pd.DataFrame(predict)
    df_gt = pd.DataFrame(gt)

    corr = df_predict.corrwith(df_gt, axis=0).values  # [F]
    return corr


def get_tops(predict, gt, tops):
    assert 0 < tops < 1
    predict = predict.reshape(-1, predict.shape[1])  # [batch_size*predict_len, num_features]
    gt = gt.reshape(-1, gt.shape[1])

    rate = np.sum(np.abs(np.abs((predict - gt)) / (gt+1e-5)) < tops, axis=0) / float(gt.shape[0])
    return rate


def get_avg_error(predict, gt):
    predict = predict.reshape(-1, predict.shape[1])  # [batch_size*predict_len, num_features]
    gt = gt.reshape(-1, gt.shape[1])

    error_rate = np.abs(predict - gt) / (gt+1e-5)
    avg_error = np.mean(error_rate, axis=0)

    return avg_error


if __name__ == '__main__':
    _predict = torch.rand(32, 3, 1024)
    _gt = torch.rand(32, 3, 1024)
    me = metrics()
    me(_predict, _gt)
