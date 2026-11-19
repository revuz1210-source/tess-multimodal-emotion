# Multimodal Emotion Recognition on TESS (speech / text / fusion)

Recognises 7 emotions (angry, disgust, fear, happy, neutral, pleasant surprise, sad) from
**speech only**, **text only**, and **speech + text (fusion)** on the Toronto Emotional Speech Set.

## Project layout
```
project/
├── common/                  shared data, model, training code
├── models/
│   ├── speech_pipeline/  train.py  test.py
│   ├── text_pipeline/    train.py  test.py
│   └── fusion_pipeline/  train.py  test.py
├── Results/                 accuracy tables + plots (written by the scripts)
├── analysis.py              tables, per-class F1, t-SNE, failure cases
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```
TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
(2,800 clips; the Kaggle copy contains each file twice, the loader de-duplicates).
Point the scripts to the folder with the wav files using `--data_root` (on Kaggle the default
`/kaggle/input` is searched automatically). The text pipeline downloads `bert-base-uncased`
from Hugging Face (internet needed once).

## Run (from the project root)
```bash
python models/speech_pipeline/train.py  && python models/speech_pipeline/test.py
python models/text_pipeline/train.py    && python models/text_pipeline/test.py
python models/fusion_pipeline/train.py  && python models/fusion_pipeline/test.py
python analysis.py
```
Useful flags: `--data_root`, `--epochs 40`, `--bs 32`, `--lr 1e-3`, `--seed 42`,
`--fusion gated|concat`, `--bert bert-base-uncased`.

## Pipeline summary
| Block | Speech | Text |
|---|---|---|
| Preprocessing | resample 16 kHz, trim silence (top_db=25), pad/crop to 3 s, per-utterance normalisation | lowercase, whitespace clean, WordPiece tokenisation |
| Feature extraction | 64-bin log-Mel spectrogram (25 ms window, 10 ms hop) | frozen BERT token embeddings (tokens x 768) |
| Temporal / contextual | Conv1d + 2-layer BiLSTM + attention pooling | Linear + BiLSTM + attention pooling |
| Fusion | gated fusion `g*s + (1-g)*t` (or concat+MLP) of the two 256-d vectors | |
| Classifier | MLP (256 -> 128 -> 7) | |

## Evaluation protocol
* Split by **target word** (70 / 10 / 20 % of the 200 words); both speakers and all emotions are in every split, so no word appears in two splits.
* Model selection on validation accuracy; test set used once. Seed 42.
* Metrics: accuracy, macro-F1, per-class F1, confusion matrices.

## Outputs (in `Results/`)
`accuracy_table.csv/.md`, `per_class_f1.csv`, `separability_silhouette.csv`, `failure_cases.csv`,
`<run>/confusion_matrix.png`, `<run>/training_curves.png`, `plots/tsne_*.png`.

## Note on the dataset
Every TESS transcript is the carrier phrase "Say the word ___"; the target words are the same
in all emotions, so the text carries almost no emotion information. Text-only accuracy is expected
to be close to chance (about 14 %), and fusion mainly mirrors the speech branch. See the report.

## Links
* GitHub: <add repo link>
* Drive (checkpoints / large files): <add public link>
