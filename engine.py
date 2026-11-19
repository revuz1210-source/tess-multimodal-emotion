"""Training loop, evaluation and result saving shared by all three pipelines."""
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common.data import EMOTIONS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_dev(b, device):
    return {k: v.to(device) for k, v in b.items()}


def run_epoch(model, loader, device, crit, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for b in loader:
            b = to_dev(b, device)
            logits, _ = model(b)
            loss = crit(logits, b["y"])
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * len(b["y"])
            correct += (logits.argmax(1) == b["y"]).sum().item()
            n += len(b["y"])
    return tot / n, correct / n


def fit(model, loaders, device, out_dir, epochs=40, lr=1e-3, patience=8):
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc, best_loss, bad, hist = -1.0, 1e9, 0, []
    for ep in range(1, epochs + 1):
        trl, tra = run_epoch(model, loaders["train"], device, crit, opt)
        val, vaa = run_epoch(model, loaders["val"], device, crit)
        hist.append(dict(epoch=ep, train_loss=trl, train_acc=tra, val_loss=val, val_acc=vaa))
        print(f"ep {ep:02d} | train loss {trl:.3f} acc {tra:.3f} | val loss {val:.3f} acc {vaa:.3f}")
        if vaa > best_acc or (vaa == best_acc and val < best_loss):
            best_acc, best_loss, bad = vaa, val, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break
    h = pd.DataFrame(hist)
    h.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.train_acc, label="train")
    ax[1].plot(h.epoch, h.val_acc, label="val")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    with open(os.path.join(out_dir, "best_val.json"), "w") as f:
        json.dump({"best_val_acc": best_acc}, f)
    return best_acc


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    L, Y, I, E = [], [], [], {}
    for b in loader:
        b = to_dev(b, device)
        logits, embs = model(b)
        L.append(logits.cpu())
        Y.append(b["y"].cpu())
        I.append(b["idx"].cpu())
        for k, v in embs.items():
            E.setdefault(k, []).append(v.cpu())
    return dict(logits=torch.cat(L).numpy(), y=torch.cat(Y).numpy(), idx=torch.cat(I).numpy(),
                emb={k: torch.cat(v).numpy() for k, v in E.items()})


def evaluate_and_save(name, out_dir, df, pred):
    y, logits = pred["y"], pred["logits"]
    p = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro")
    rep = classification_report(y, p, labels=range(len(EMOTIONS)), target_names=EMOTIONS,
                                output_dict=True, zero_division=0)
    best_val = None
    bv = os.path.join(out_dir, "best_val.json")
    if os.path.exists(bv):
        best_val = json.load(open(bv))["best_val_acc"]
    metrics = dict(model=name, test_acc=acc, test_macro_f1=f1, best_val_acc=best_val,
                   per_class_f1={e: rep[e]["f1-score"] for e in EMOTIONS})
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{name}] test acc {acc:.4f} | macro-F1 {f1:.4f}")
    print(classification_report(y, p, labels=range(len(EMOTIONS)), target_names=EMOTIONS, zero_division=0))

    cm = confusion_matrix(y, p, labels=range(len(EMOTIONS)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right"); ax.set_yticklabels(EMOTIONS)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name}: confusion matrix (test)")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    sub = df.iloc[pred["idx"]].reset_index(drop=True)
    sub["true"] = [EMOTIONS[i] for i in y]
    sub["pred"] = [EMOTIONS[i] for i in p]
    sub["confidence"] = probs.max(1)
    sub["correct"] = sub["true"] == sub["pred"]
    sub.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    np.save(os.path.join(out_dir, "labels.npy"), y)
    for k, v in pred["emb"].items():
        np.save(os.path.join(out_dir, f"emb_{k}.npy"), v)
    return metrics
