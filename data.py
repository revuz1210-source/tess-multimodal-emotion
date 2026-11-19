"""Data loading, preprocessing and feature extraction for TESS."""
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}

SR = 16000        # sampling rate (Hz)
MAX_SEC = 3.0     # clips are padded / cropped to this length
N_MELS = 64
N_FFT = 512
HOP = 160         # 10 ms hop -> 100 frames / second
TOP_DB = 25       # silence-trimming threshold
MAX_TOK = 12      # max BERT tokens ("[CLS] say the word X [SEP]" is ~6)


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
def clean_text(s):
    """Text preprocessing: lowercase + collapse whitespace."""
    return " ".join(str(s).lower().strip().split())


def build_dataframe(root=None):
    """Scan the TESS wav files and build one row per unique (speaker, word, emotion).

    The Kaggle copy of TESS contains every file twice (nested folder), so we
    de-duplicate. Transcript = 'Say the word <word>' (TESS carrier phrase).
    """
    root = root or os.environ.get("TESS_ROOT", "/kaggle/input")
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    rows = []
    for f in files:
        parts = os.path.basename(f)[:-4].split("_")
        if len(parts) != 3:
            continue
        spk, word, emo = parts
        spk = spk.upper()
        if spk.startswith("OA"):      # one file is named 'OA_...' (typo in dataset)
            spk = "OAF"
        elif spk.startswith("YA"):
            spk = "YAF"
        emo = emo.lower()
        if emo in ("ps", "pleasant", "pleasant_surprised"):
            emo = "pleasant_surprise"
        if emo not in EMO2ID:
            continue
        rows.append(dict(path=f, speaker=spk, word=word.lower(), emotion=emo))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No TESS wav files found under {root}. Set --data_root.")
    df = df.drop_duplicates(subset=["speaker", "word", "emotion"]).reset_index(drop=True)
    df["text"] = df["word"].map(lambda w: clean_text(f"Say the word {w}"))
    return df


def add_splits(df, seed=42, test_frac=0.2, val_frac=0.1):
    """Word-disjoint split: a target word never appears in two splits.

    Both speakers and all emotions appear in every split. This avoids the
    leakage you get from a random split (same word + same speaker in train/test).
    """
    words = sorted(df["word"].unique())
    random.Random(seed).shuffle(words)
    n_test, n_val = int(len(words) * test_frac), int(len(words) * val_frac)
    test = set(words[:n_test])
    val = set(words[n_test:n_test + n_val])
    df = df.copy()
    df["split"] = df["word"].map(lambda w: "test" if w in test else "val" if w in val else "train")
    return df


# ----------------------------------------------------------------------------
# Speech: preprocessing + log-Mel features
# ----------------------------------------------------------------------------
def extract_mel(path):
    import librosa
    y, _ = librosa.load(path, sr=SR)                       # resample to 16 kHz mono
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)           # trim leading/trailing silence
    size = int(SR * MAX_SEC)
    n_valid = min(len(y), size)
    y = librosa.util.fix_length(y, size=size)               # pad / crop
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    m = librosa.power_to_db(m, ref=np.max).T                # (time, n_mels)
    n_frames = max(1, min(m.shape[0], n_valid // HOP + 1))
    valid = m[:n_frames]
    m = (m - valid.mean()) / (valid.std() + 1e-6)           # per-utterance normalisation
    m[n_frames:] = 0.0
    return m.astype(np.float32), n_frames


def get_speech_features(df, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, "speech_mel.npz")
    paths = df["path"].tolist()
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if list(z["paths"]) == paths:
            return z["mel"], z["lens"]
    mels, lens = [], []
    for p in tqdm(paths, desc="log-Mel"):
        m, n = extract_mel(p)
        mels.append(m)
        lens.append(n)
    mel, lens = np.stack(mels), np.array(lens, dtype=np.int64)
    np.savez(f, mel=mel, lens=lens, paths=np.array(paths, dtype=object))
    return mel, lens


# ----------------------------------------------------------------------------
# Text: tokenisation + frozen BERT token features
# ----------------------------------------------------------------------------
def get_text_features(df, cache_dir, bert="bert-base-uncased"):
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, f"text_{bert.replace('/', '_')}.npz")
    texts = df["text"].tolist()
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if list(z["texts"]) == texts:
            return z["tok"].astype(np.float32), z["mask"]
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(bert)
    model = AutoModel.from_pretrained(bert).to(dev).eval()
    feats, masks = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 64), desc="BERT"):
            enc = tok(texts[i:i + 64], padding="max_length", truncation=True,
                      max_length=MAX_TOK, return_tensors="pt").to(dev)
            feats.append(model(**enc).last_hidden_state.cpu().numpy())
            masks.append(enc["attention_mask"].cpu().numpy())
    tokf, mask = np.concatenate(feats), np.concatenate(masks).astype(np.int64)
    np.savez(f, tok=tokf.astype(np.float16), mask=mask, texts=np.array(texts, dtype=object))
    return tokf.astype(np.float32), mask


# ----------------------------------------------------------------------------
# Torch datasets
# ----------------------------------------------------------------------------
class DictDataset(Dataset):
    def __init__(self, tensors):
        self.t = tensors
        self.n = len(next(iter(tensors.values())))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.t.items()}


def make_loaders(df, feats, bs=32, seed=42):
    y = df["emotion"].map(EMO2ID).values
    loaders = {}
    for s in ["train", "val", "test"]:
        idx = np.where(df["split"].values == s)[0]
        t = {"y": torch.tensor(y[idx], dtype=torch.long), "idx": torch.tensor(idx, dtype=torch.long)}
        for k, v in feats.items():
            t[k] = torch.from_numpy(np.ascontiguousarray(v[idx]))
        g = torch.Generator()
        g.manual_seed(seed)
        loaders[s] = DataLoader(DictDataset(t), batch_size=bs, shuffle=(s == "train"), generator=g)
    return loaders
