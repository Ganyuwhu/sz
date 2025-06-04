import torch


class TriangularCausalMask:
    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]

        # 构造一个上三角掩码使得未来时间步上的信息被屏蔽
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores: torch.Tensor, device='cpu'):
        """
        :param B: batch size
        :param H: heads
        :param L: sequence length
        :param index: choose which positions should be masked
        :param scores: attention score
        :param device:
        """

        _mask = torch.ones((L, scores.shape[-1]), dtype=torch.bool).to(device).triu(1)  # (L, scores.shape[-1])
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])  # (B, H, L, scores.shape[-1])

        # torch.expand本质上是对返回一个对原张量进行广播操作后得到的一个视图，因此改变_mask_ex会影响到_mask里的数据

        # 创建提取器
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :].to(device)

        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
