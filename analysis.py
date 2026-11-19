"""Builds the accuracy table, per-class comparison, t-SNE plots and failure cases.

Run AFTER train.py + test.py of all three pipelines:   python analysis.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import json

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
ROOT = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--results", default=os.path.join(ROOT, "Results"))
args = p.parse_args()
R = args.results
PLOTS = os.path.join(R, "plots")
os.makedirs(PLOTS, exist_ok=True)
RUNS = [r for r in ["speech", "text", "fusion"] if os.path.exists(os.path.join(R, r, "metrics.json"))]

# ---- 1. accuracy table -------------------------------------------------------
rows, per_class = [], {}
for r in RUNS:
    m = json.load(open(os.path.join(R, r, "metrics.json")))
    rows.append({"model": r, "val_acc": m["best_val_acc"], "test_acc": m["test_acc"],
                 "test_macro_f1": m["test_macro_f1"]})
    per_class[r] = m["per_class_f1"]
tab = pd.DataFrame(rows).round(4)
tab.to_csv(os.path.join(R, "accuracy_table.csv"), index=False)
with open(os.path.join(R, "accuracy_table.md"), "w") as f:
    f.write(tab.to_markdown(index=False) if hasattr(tab, "to_markdown") else tab.to_string(index=False))
print(tab.to_string(index=False))

# ---- 2. per-class F1 comparison ---------------------------------------------
pc = pd.DataFrame(per_class).round(4)
pc.to_csv(os.path.join(R, "per_class_f1.csv"))
print(pc)
ax = pc.plot(kind="bar", figsize=(9, 4))
ax.set_ylabel("F1 (test)"); ax.set_title("Per-emotion F1: speech vs text vs fusion")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "per_class_f1.png"), dpi=150); plt.close()

# ---- 3. t-SNE / separability of learned representations ---------------------
TITLES = {"temporal": "Temporal modelling block", "contextual": "Contextual modelling block",
          "fusion": "Fusion block"}
sep, cache = [], {}
for r in RUNS:
    y = np.load(os.path.join(R, r, "labels.npy"))
    for k in TITLES:
        f = os.path.join(R, r, f"emb_{k}.npy")
        if not os.path.exists(f):
            continue
        X = np.load(f)
        Z = TSNE(n_components=2, perplexity=min(30, len(X) // 4), init="pca", random_state=0).fit_transform(X)
        cache[(r, k)] = (Z, y)
        sep.append({"run": r, "block": k, "silhouette": round(float(silhouette_score(X, y)), 4)})
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for i, e in enumerate(EMOTIONS):
            ax.scatter(*Z[y == i].T, s=10, label=e)
        ax.set_title(f"{r} run: {TITLES[k]}"); ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, f"tsne_{r}_{k}.png"), dpi=150); plt.close()
if sep:
    pd.DataFrame(sep).to_csv(os.path.join(R, "separability_silhouette.csv"), index=False)
    print(pd.DataFrame(sep))
panel = [(r, k) for (r, k) in [("fusion", "temporal"), ("fusion", "contextual"), ("fusion", "fusion")] if (r, k) in cache]
if panel:
    fig, axs = plt.subplots(1, len(panel), figsize=(5 * len(panel), 4.8))
    axs = np.atleast_1d(axs)
    for ax, (r, k) in zip(axs, panel):
        Z, y = cache[(r, k)]
        for i, e in enumerate(EMOTIONS):
            ax.scatter(*Z[y == i].T, s=8, label=e)
        ax.set_title(TITLES[k]); ax.set_xticks([]); ax.set_yticks([])
    axs[0].legend(fontsize=7, markerscale=1.5)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "tsne_three_blocks.png"), dpi=150); plt.close()

# ---- 4. failure cases (most confident mistakes) ------------------------------
out = []
for r in RUNS:
    d = pd.read_csv(os.path.join(R, r, "predictions.csv"))
    bad = d[~d.correct].sort_values("confidence", ascending=False).head(5)
    bad = bad.assign(model=r)[["model", "speaker", "word", "text", "true", "pred", "confidence", "path"]]
    out.append(bad)
if out:
    fc = pd.concat(out); fc["confidence"] = fc["confidence"].round(3)
    fc.to_csv(os.path.join(R, "failure_cases.csv"), index=False)
    print(fc.drop(columns="path").to_string(index=False))
print("\nSaved tables and plots to", R)
