import torch
import torch.nn as nn
import torch.functional as f
from torch import Tensor
from typing import Optional

from utils.Layer_Tools import positional_encoding, Transpose, get_activation_fn, normalize_tensor, GateLayer, \
    moving_avg, series_decomp, MultiheadAttention, Flatten_Head


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, norm='LayerNorm', attn_dropout=0., dropout=0.1,
                 activation="gelu", res_attention=False, pre_norm=False):
        """
        :param d_model: 输入向量最后一个维度的元素数
        :param n_heads: 多头注意力的头的个数
        :param d_k: 键的维度
        :param d_v: 值的维度
        :param d_ff: ff网络中间层的温度
        :param norm: 归一化方法
        :param attn_dropout: 注意力模块的dropout值
        :param dropout: ff网络的dropout值
        :param activation: 激活函数类型
        :param res_attention: 是否保留当前层的注意力得分
        :param pre_norm: 是否在执行计算前进行归一化
        """

        super().__init__()

        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.time_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                            proj_dropout=dropout, res_attention=res_attention)
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                            proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=True),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=True))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm

    def forward(self, src: Tensor, time_stamp, prev: Optional[Tensor] = None, time_prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        # src: [B, N, L, C]
        B, N, L, C = src.shape

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        # Time-Attention
        time_scores = torch.empty(1)
        if self.res_attention:
            src, time_attn, time_scores = self.time_attn(src, time_stamp, time_stamp, time_prev,
                                                         key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
        else:
            src, time_attn = self.time_attn(src, time_stamp, time_stamp,
                                            key_padding_mask=key_padding_mask,
                                            attn_mask=attn_mask
                                            )

        # Multi-Head attention
        scores = torch.empty(1)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        src = src.reshape((B, N * L, C))
        # Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        src2 = self.ff(src)

        # Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        src = src.reshape(B, N, L, C)

        if self.res_attention:
            return src, time_scores, scores
        else:
            return src


class TSTEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_k=None, d_v=None, d_ff=256, norm='LayerNorm', attn_dropout=0.,
                 dropout=0.1, activation="gelu", res_attention=False, pre_norm=False):
        """
        :param n_layers: Encoder Layer的个数，其他同上
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(d_model, n_heads, d_k, d_v, d_ff, norm, attn_dropout, dropout, activation, res_attention,
                             pre_norm)
             for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src, time_stamp, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        time_scores = None

        if self.res_attention:
            for mod in self.layers:
                output, time_scores, scores = mod(output, time_stamp, prev=scores, time_prev=time_scores,
                                                  key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, time_stamp, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTiEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, pe='zeros', learn_pe=True, n_layers=3, d_model=128, n_heads=16, d_k=None,
                 d_v=None, d_ff=256, norm='LayerNorm', attn_dropout=0., dropout=0., activation="gelu",
                 res_attention=True, pre_norm=False):
        """
        :param patch_num: patch的数目
        :param patch_len: 每一个patch中包含的时间步数
        :param pe: positional encoding方法
        :param learn_pe: pe是否可学习
        其他的同上
        """

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_time = nn.Linear(patch_len, d_model)

        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        self.res_attention = res_attention

        # Encoder layers
        self.encoder = TSTEncoder(n_layers, d_model, n_heads, d_k, d_v, d_ff, norm, attn_dropout, dropout, activation,
                                  res_attention, pre_norm)

    def forward(self, x, time_stamp):
        # x: [bs x nvars x patch_len x patch_num]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        time_stamp = time_stamp.permute(0, 1, 3, 2)

        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
        time_stamp = self.W_time(time_stamp)
        x = x + self.W_pos  # 加上位置编码
        x = self.dropout(x)

        # Encoder layers
        x = self.encoder(x, time_stamp)

        # 最后调整维度
        x = x.permute(0, 1, 3, 2)  # [bs x n_vars x d_model x seq_len]
        return x


class TSTDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, norm='LayerNorm', attn_dropout=0., dropout=0.1,
                 activation="gelu", res_attention=False, pre_norm=False):
        """
        :param d_model: 输入向量最后一个维度的元素数
        :param n_heads: 多头注意力的头的个数
        :param d_k: 键的维度
        :param d_v: 值的维度
        :param d_ff: ff网络中间层的温度
        :param norm: 归一化方法
        :param attn_dropout: 注意力模块的dropout值
        :param dropout: ff网络的dropout值
        :param activation: 激活函数类型
        :param res_attention: 是否保留当前层的注意力得分
        :param pre_norm: 是否在执行计算前进行归一化
        """
        super().__init__()

        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.time_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                            proj_dropout=dropout, res_attention=res_attention)
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                            proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=True),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=True))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm

    def forward(self, src: Tensor, encoder_output: Tensor, time_stamp, prev: Optional[Tensor] = None,
                time_prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask:
                Optional[Tensor] = None):

        # src: [B, N, L, C]
        B, N, L, C = src.shape
        pass




class patchTST2(nn.Module):
    def __init__(self, context_window: int, target_window: int, patch_len: int, stride: int,
                 pe: str = 'zeros', learn_pe: bool = True, n_layers: int = 3, d_model: int = 128,
                 n_heads: int = 16, d_k: Optional[int] = None, d_v: Optional[int] = None, d_ff: int = 256,
                 norm: str = 'LayerNorm', attn_dropout: float = 0., dropout: float = 0., activation: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, pretrain_head: bool = False,
                 head_type='flatten', target: Optional[list] = None, memory: Optional[list] = None):

        super().__init__()
        pass

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )

