"""Model blocks: speech temporal model, text contextual model, fusion, classifier."""
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_CLASSES = 7


def spec_augment(x, f_max=8, t_max=20):
    """Light SpecAugment (random frequency + time masks) for training only."""
    B, T, F = x.shape
    x = x.clone()
    for b in range(B):
        f = random.randint(0, f_max)
        f0 = random.randint(0, F - f)
        x[b, :, f0:f0 + f] = 0
        t = random.randint(0, t_max)
        t0 = random.randint(0, max(0, T - t))
        x[b, t0:t0 + t, :] = 0
    return x


class AttnPool(nn.Module):
    """Attention pooling over time / tokens (padding masked out)."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x, mask):
        s = self.w(x).squeeze(-1).masked_fill(~mask, -1e9)
        a = s.softmax(-1).unsqueeze(-1)
        return (a * x).sum(1)


def run_bilstm(lstm, x, lens):
    T = x.size(1)
    packed = pack_padded_sequence(x, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
    return out


class SpeechEncoder(nn.Module):
    """Temporal modelling: Conv1d front-end -> 2-layer BiLSTM -> attention pooling."""

    def __init__(self, n_mels=64, hidden=128, out=256, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_mels, 128, 5, padding=2), nn.BatchNorm1d(128),
                                  nn.ReLU(), nn.Dropout(drop))
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.pool = AttnPool(2 * hidden)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, out), nn.ReLU(), nn.Dropout(drop))

    def forward(self, mel, lens):
        x = spec_augment(mel) if self.training else mel
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = run_bilstm(self.lstm, x, lens)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]
        return self.proj(self.pool(x, mask))


class TextEncoder(nn.Module):
    """Contextual modelling: BERT token features -> BiLSTM -> attention pooling."""

    def __init__(self, in_dim=768, hidden=128, out=256, drop=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(drop))
        self.lstm = nn.LSTM(256, hidden, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * hidden)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, out), nn.ReLU(), nn.Dropout(drop))

    def forward(self, tok, mask):
        mask = mask.bool()
        x = self.inp(tok)
        x = run_bilstm(self.lstm, x, mask.sum(1))
        return self.proj(self.pool(x, mask))


def make_classifier(d, drop=0.3):
    return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, N_CLASSES))


class SpeechModel(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.enc = SpeechEncoder()
        self.clf = make_classifier(256)

    def forward(self, b):
        e = self.enc(b["mel"], b["mel_len"])
        return self.clf(e), {"temporal": e}


class TextModel(nn.Module):
    def __init__(self, text_dim=768, **kw):
        super().__init__()
        self.enc = TextEncoder(in_dim=text_dim)
        self.clf = make_classifier(256)

    def forward(self, b):
        e = self.enc(b["tok"], b["tok_mask"])
        return self.clf(e), {"contextual": e}


class FusionModel(nn.Module):
    """Speech + text fusion. 'gated': f = g*s + (1-g)*t, g = sigmoid(W[s;t]).
    'concat': f = ReLU(W[s;t])."""

    def __init__(self, text_dim=768, fusion="gated", **kw):
        super().__init__()
        self.fusion = fusion
        self.speech = SpeechEncoder()
        self.text = TextEncoder(in_dim=text_dim)
        if fusion == "gated":
            self.gate = nn.Linear(512, 256)
        else:
            self.mix = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.clf = make_classifier(256)

    def forward(self, b):
        s = self.speech(b["mel"], b["mel_len"])
        t = self.text(b["tok"], b["tok_mask"])
        cat = torch.cat([s, t], dim=-1)
        if self.fusion == "gated":
            g = torch.sigmoid(self.gate(cat))
            f = g * s + (1 - g) * t
        else:
            f = self.mix(cat)
        return self.clf(f), {"temporal": s, "contextual": t, "fusion": f}


def build_model(kind, text_dim=768, fusion="gated"):
    if kind == "speech":
        return SpeechModel()
    if kind == "text":
        return TextModel(text_dim=text_dim)
    return FusionModel(text_dim=text_dim, fusion=fusion)
