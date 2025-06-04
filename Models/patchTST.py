from torch import nn
from Layers.patchTST import patchTST


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.model = patchTST(
            context_window=configs.context_window,
            target_window=configs.target_window,
            patch_len=configs.patch_len,
            stride=configs.stride,
            pe=configs.pe,
            learn_pe=configs.learn_pe,
            n_layers=configs.n_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_k=configs.d_k,
            d_v=configs.d_v,
            d_ff=configs.d_ff,
            norm=configs.norm,
            attn_dropout=configs.attn_dropout,
            dropout=configs.attn_dropout,
            activation=configs.activation,
            res_attention=configs.res_attention,
            pre_norm=configs.pre_norm,
            pretrain_head=configs.pretrain_head,
            head_type=configs.head_type,
            target=configs.target
        )

    def forward(self, x, time_stamp):
        return self.model(x, time_stamp)

