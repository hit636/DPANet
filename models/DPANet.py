import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------- configs-----------------
class Configs:
    """
    用于存储 DPA-Net 模型超参数的配置类
    """
    def __init__(self, seq_len=96, pred_len=96, enc_in=7, 
                 num_scales=3, d_model=128, n_heads=8,
                 d_ff=256, dropout=0.1, revin=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in      # C: 通道数
        self.num_scales = num_scales   # S: 金字塔的层数
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.revin = revin

# ----------------- RevIN -----------------
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features; self.eps = eps; self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def forward(self, x, mode: str):
        if mode == 'norm': self._get_statistics(x); x = self._normalize(x)
        elif mode == 'denorm': x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
    def _normalize(self, x):
        x = x - self.mean; x = x / self.stdev
        if self.affine: x = x * self.affine_weight + self.affine_bias
        return x
    def _denormalize(self, x):
        if self.affine: x = (x - self.affine_bias) / (self.affine_weight + 1e-8)
        x = x * self.stdev + self.mean
        return x

# ----------------- CrossPyramidFusionBlock -----------------
class CrossPyramidFusionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(CrossPyramidFusionBlock, self).__init__()
        self.cross_attn_t_to_f = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_f_to_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model * 2)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_t, h_f):
        # h_t, h_f: [B*C, L_s, d_model]
        
        # cross_attetion
        f_from_t, _ = self.cross_attn_t_to_f(query=h_t, key=h_f, value=h_f)
        h_t = self.norm1(h_t + self.dropout(f_from_t))
        
        t_from_f, _ = self.cross_attn_f_to_t(query=h_f, key=h_t, value=h_t)
        h_f = self.norm2(h_f + self.dropout(t_from_f))
        
        # FFN
        fused = torch.cat([h_t, h_f], dim=-1)
        fused = self.norm3(fused + self.dropout(self.ffn(fused)))
        
        # 分离
        h_t_out, h_f_out = torch.chunk(fused, 2, dim=-1)
        return h_t_out, h_f_out

# ----------------- DP-Net -----------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        configs.revin=True
        configs.num_scales= 3
        configs.d_model=128
        configs.n_heads=8
        self.configs = configs
        
        if configs.revin:
            self.revin_layer = RevIN(configs.enc_in)

        # --- temporal pyramid ---
        self.temporal_downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2, stride=2) for _ in range(configs.num_scales - 1)
        ])
        
        # --- feature embeding ---
        self.temporal_embedders = nn.ModuleList()
        self.spectral_embedders = nn.ModuleList()
        
        for i in range(configs.num_scales):
            # 动态计算每个时域尺度的长度
            temporal_len_s = configs.seq_len // (2**i)
            self.temporal_embedders.append(nn.Linear(temporal_len_s, configs.d_model))
            
            # 频域子信号的长度与原始输入长度相同
            spectral_len_s = configs.seq_len
            self.spectral_embedders.append(nn.Linear(spectral_len_s, configs.d_model))

        # --- 跨金字塔融合块 ---
        self.fusion_blocks = nn.ModuleList([
            CrossPyramidFusionBlock(configs.d_model, configs.n_heads, configs.d_ff, configs.dropout)
            for _ in range(configs.num_scales)
        ])
        
        # --- 预测头 ---
        self.prediction_head = nn.Linear(configs.d_model, configs.pred_len)

    def get_frequency_bands(self, length, num_scales):
        freqs = np.fft.rfftfreq(length)
        if num_scales == 1: return [(0, len(freqs))]
        log_max_freq = np.log(len(freqs)) if len(freqs) > 1 else 1
        log_bounds = np.linspace(0, log_max_freq, num_scales + 1)
        bounds = np.exp(log_bounds).astype(int); bounds[0] = 0; bounds[-1] = len(freqs)
        bands = []
        for i in range(num_scales):
            start, end = bounds[i], bounds[i+1]
            if start >= end and i > 0: start = bands[-1][1]
            if start >= len(freqs): start = len(freqs) - 1
            end = min(end, len(freqs))
            bands.append((start, end))
        return bands

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L_in, C = x_enc.shape
        
        if self.configs.revin:
            x_enc = self.revin_layer(x_enc, 'norm')

        # 阶段一: 构建双金字塔
        temporal_pyramid = [x_enc]
        for i in range(self.configs.num_scales - 1):
            downsampled = self.temporal_downsamplers[i](temporal_pyramid[-1].permute(0, 2, 1)).permute(0, 2, 1)
            temporal_pyramid.append(downsampled)

        x_fft = torch.fft.rfft(x_enc, dim=1)
        bands = self.get_frequency_bands(L_in, self.configs.num_scales)
        spectral_pyramid = []
        for low_idx, high_idx in bands:
            temp_fft = torch.zeros_like(x_fft)
            if low_idx < high_idx:
                temp_fft[:, low_idx:high_idx, :] = x_fft[:, low_idx:high_idx, :]
            sub_signal = torch.fft.irfft(temp_fft, n=L_in, dim=1)
            spectral_pyramid.append(sub_signal)
        
        # 阶段二: 金字塔特征嵌入
        temporal_pyramid_flat = [t.permute(0, 2, 1).reshape(B*C, -1) for t in temporal_pyramid]
        spectral_pyramid_flat = [s.permute(0, 2, 1).reshape(B*C, -1) for s in spectral_pyramid]

        h_t_pyramid = [self.temporal_embedders[i](temporal_pyramid_flat[i]) for i in range(self.configs.num_scales)]
        h_f_pyramid = [self.spectral_embedders[i](spectral_pyramid_flat[i]) for i in range(self.configs.num_scales)]
        
        h_t_pyramid = [h.unsqueeze(1) for h in h_t_pyramid]
        h_f_pyramid = [h.unsqueeze(1) for h in h_f_pyramid]

        # 阶段三: 跨金字塔融合 (由粗到细)
        h_t_coarse, h_f_coarse = self.fusion_blocks[-1](h_t_pyramid[-1], h_f_pyramid[-1])
        
        for s in range(self.configs.num_scales - 2, -1, -1):
            # 上采样时需要注意序列长度可能为1
            if h_t_coarse.shape[1] > 1:
                h_t_coarse_up = F.interpolate(h_t_coarse.permute(0, 2, 1), size=h_t_pyramid[s].shape[1], mode='linear').permute(0, 2, 1)
                h_f_coarse_up = F.interpolate(h_f_coarse.permute(0, 2, 1), size=h_f_pyramid[s].shape[1], mode='linear').permute(0, 2, 1)
            else: # 如果长度为1，直接复制
                h_t_coarse_up = h_t_coarse.repeat(1, h_t_pyramid[s].shape[1], 1)
                h_f_coarse_up = h_f_coarse.repeat(1, h_f_pyramid[s].shape[1], 1)

            h_t_current = h_t_pyramid[s] + h_t_coarse_up
            h_f_current = h_f_pyramid[s] + h_f_coarse_up
            
            h_t_coarse, h_f_coarse = self.fusion_blocks[s](h_t_current, h_f_current)

        # 阶段四: 预测头
        final_repr = h_t_coarse.squeeze(1)
        prediction = self.prediction_head(final_repr)
        
        prediction = prediction.view(B, C, self.configs.pred_len).permute(0, 2, 1)

        if self.configs.revin:
            prediction = self.revin_layer(prediction, 'denorm')
            
        return prediction