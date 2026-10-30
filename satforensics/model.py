import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.lin = nn.Linear(d_in, d_model)

    def forward(self, x):
        return self.lin(self.norm(x))


class GatedFusion(nn.Module):
    def __init__(self, d_model, n_mod):
        super().__init__()
        self.n_mod = n_mod
        self.gate = nn.Linear(n_mod * d_model, n_mod)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, zs):
        g = torch.sigmoid(self.gate(torch.cat(zs, dim=-1)))
        mix = sum(g[..., i:i + 1] * zs[i] for i in range(self.n_mod))
        return F.relu(self.out(mix)), g


class ConcatFusion(nn.Module):
    def __init__(self, d_model, n_mod):
        super().__init__()
        self.out = nn.Linear(n_mod * d_model, d_model)

    def forward(self, zs):
        return F.relu(self.out(torch.cat(zs, dim=-1))), None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class SatForensics(nn.Module):
    def __init__(self, dims, n_classes=5, d_model=64, layers=2, heads=4, d_ff=128, dropout=0.1,
                 gated=True, use_pe=True):
        super().__init__()
        self.dims = list(dims)
        self.proj = nn.ModuleList([Projector(d, d_model) for d in self.dims])
        self.fusion = GatedFusion(d_model, len(self.dims)) if gated and len(self.dims) > 1 else None
        if self.fusion is None and len(self.dims) > 1:
            self.fusion = ConcatFusion(d_model, len(self.dims))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pe = PositionalEncoding(d_model) if use_pe else None
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.det = nn.Linear(d_model, 1)
        self.attr = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, mods):
        zs = [p(m) for p, m in zip(self.proj, mods)]
        gates = None
        if self.fusion is None:
            x = F.relu(zs[0])
        else:
            x, gates = self.fusion(zs)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        if self.pe is not None:
            x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        h = self.norm(x)[:, 0]
        return self.det(h).squeeze(-1), self.attr(h), zs, h, gates


def info_nce(zs, h, temperature=0.1):
    hn = F.normalize(h, dim=-1)
    loss = 0.0
    target = torch.arange(h.size(0), device=h.device)
    for z in zs:
        zn = F.normalize(z.mean(dim=1), dim=-1)
        logits = zn @ hn.t() / temperature
        loss = loss + F.cross_entropy(logits, target)
    return loss / len(zs)


class CNN1D(nn.Module):
    def __init__(self, d_in, n_classes=5, channels=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.net = nn.Sequential(
            nn.Conv1d(d_in, channels, 5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.det = nn.Linear(channels, 1)
        self.attr = nn.Linear(channels, n_classes)

    def forward(self, x):
        h = self.net(self.norm(x).transpose(1, 2)).squeeze(-1)
        return self.det(h).squeeze(-1), self.attr(h)


class LSTMNet(nn.Module):
    def __init__(self, d_in, n_classes=5, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.rnn = nn.LSTM(d_in, hidden, num_layers=2, batch_first=True)
        self.det = nn.Linear(hidden, 1)
        self.attr = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.rnn(self.norm(x))
        h = out[:, -1]
        return self.det(h).squeeze(-1), self.attr(h)
