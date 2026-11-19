"""Shared train/test entry points used by every pipeline's train.py / test.py."""
import argparse
import os

import torch

from common.data import add_splits, build_dataframe, get_speech_features, get_text_features, make_loaders
from common.engine import evaluate_and_save, fit, predict, set_seed
from common.models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=None, help="folder containing the TESS wav files")
    p.add_argument("--results", default=os.path.join(ROOT, "Results"))
    p.add_argument("--cache", default="/kaggle/working/cache" if os.path.isdir("/kaggle/working")
                   else os.path.join(ROOT, "cache"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bert", default="bert-base-uncased")
    p.add_argument("--fusion", default="gated", choices=["gated", "concat"])
    return p.parse_args()


def prepare(kind, args):
    df = add_splits(build_dataframe(args.data_root), args.seed)
    print(df["split"].value_counts().to_dict())
    feats = {}
    if kind in ("speech", "fusion"):
        feats["mel"], feats["mel_len"] = get_speech_features(df, args.cache)
    if kind in ("text", "fusion"):
        feats["tok"], feats["tok_mask"] = get_text_features(df, args.cache, args.bert)
    text_dim = feats["tok"].shape[-1] if "tok" in feats else 768
    return df, make_loaders(df, feats, args.bs, args.seed), text_dim


def train_main(kind):
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, loaders, text_dim = prepare(kind, args)
    model = build_model(kind, text_dim, args.fusion).to(device)
    print(f"{kind}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device={device}")
    out = os.path.join(args.results, kind)
    fit(model, loaders, device, out, args.epochs, args.lr, args.patience)
    print(f"Saved best checkpoint to {out}/best.pt. Now run the matching test.py")


def test_main(kind):
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, loaders, text_dim = prepare(kind, args)
    model = build_model(kind, text_dim, args.fusion).to(device)
    out = os.path.join(args.results, kind)
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    pred = predict(model, loaders["test"], device)
    evaluate_and_save(kind, out, df, pred)
