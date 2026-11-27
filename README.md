# HAM10000 Skin Lesion Classification

7-class skin lesion classification on HAM10000 with TensorFlow/Keras: a custom CNN compared with VGG16 / ResNet-50 transfer learning, with lesion-level splits that prevent data leakage, and a full error analysis.

**Course:** AIE231 Neural Networks, Faculty of Computer Science & Engineering, Alamein International University
**Status:** Phase 1 complete (dataset checked, training time estimated, versions pinned). Next: Phase 2 — EDA, preprocessing and lesion-level split. Training, evaluation and demo commands will be added here as each phase is completed.

---

## Team

| Name | GitHub | Work package |
|------|--------|--------------|
| Rohima Ahmed Ibrahim | [rohimaahmed101-sys](https://github.com/rohimaahmed101-sys) | To be assigned |
| Shaimaa Kotit | [shaimaakotit-29](https://github.com/shaimaakotit-29) | To be assigned |
| Doha Mahmoud | [dohaalaraby-lang](https://github.com/dohaalaraby-lang) | To be assigned |
|Radwa Sabry Saber | [RadwaSabry1](https://github.com/RadwaSabry1) | To be assigned|

---

## Problem statement

Clinicians who review dermatoscopic images of pigmented skin lesions need help deciding which lesions deserve urgent specialist attention. Our model takes one dermatoscopic image and predicts which of seven diagnostic categories it belongs to (actinic keratoses / intraepithelial carcinoma, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanoma, melanocytic nevi, vascular lesions), with a probability for each class. The most costly error is a missed malignancy: labelling a melanoma, basal cell carcinoma or actinic keratosis as a benign lesion can delay treatment, while flagging a benign lesion as suspicious costs only an extra specialist review. The model is a course project that illustrates decision support; **it is not a diagnostic tool.**

## Task definition

| Element | Decision | Justification |
|---|---|---|
| Input | One RGB dermatoscopic image, resized to 224 × 224 × 3 | Native input size of VGG16 and ResNet-50, so all models see the same input. All originals are 600 × 450 RGB |
| Pixel scaling | Custom CNN: pixels / 255. VGG16 / ResNet-50: the model's own Keras `preprocess_input` | Pretrained weights expect the scaling used during ImageNet training |
| Output | Dense(7) with softmax | Classes are mutually exclusive; one probability per class. Prediction = argmax, confidence = its probability |
| Loss | Categorical cross-entropy | Standard pairing with softmax for single-label multiclass problems. A class-weighted version is tested for imbalance |
| Primary metric | **Macro-F1** on the validation set (used for model selection) | Every class counts equally; 66.95 % of images are `nv`, so accuracy alone is misleading |
| Key secondary metric | **Melanoma recall** | A missed melanoma is the most dangerous error |
| Also reported | Accuracy, confusion matrix, per-class precision / recall / F1, one-vs-rest ROC and AUC | Required on the test set for the final model and both baselines |

## Dataset

- **Name:** HAM10000 ("Human Against Machine with 10000 training images")
- **Download used:** Kaggle mirror [`kmader/skin-cancer-mnist-ham10000`](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Licence:** CC BY-NC 4.0 (non-commercial use with attribution), which permits academic use
- **Citation:** Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

Raw data is **not** stored in this repository. See [`data/README.md`](data/README.md) to download it.

### Dataset check (Phase 1)

Measured by [`notebooks/00_phase1_dataset_check.ipynb`](notebooks/00_phase1_dataset_check.ipynb) on 20 Nov 2026:

| Finding | Value |
|---|---|
| Unique images | 10,015 (the Kaggle copy stores each image twice; all copies are identical) |
| Classes | 7 — nv 6,705 (66.95 %), mel 1,113 (11.11 %), bkl 1,099 (10.97 %), bcc 514 (5.13 %), akiec 327 (3.27 %), vasc 142 (1.42 %), df 115 (1.15 %) |
| Imbalance ratio (largest / smallest) | 58.3 |
| Lesions | 7,470; 1,956 lesions have 2–6 images (4,501 images) → splits are made by `lesion_id` |
| Byte-identical images with different IDs | 2 pairs |
| Missing metadata | age 57 NaN; sex 57 "unknown"; localization 234 "unknown" |
| Image size | All 600 × 450 RGB |

### Training-time check (Phase 1, estimate)

Measured by [`notebooks/00b_training_time_check.ipynb`](notebooks/00b_training_time_check.ipynb) on a free Colab Tesla T4 (TensorFlow 2.21.0), 224 × 224 images, batch 32. The spec requires one training run to finish in under 30 minutes.

| Model (placeholder for timing) | Estimated 30-epoch run | Max epochs in 30 min |
|---|---|---|
| A Custom CNN | 6.4 min | 167 |
| B ResNet-50 frozen | 16.9 min | 53 |
| C ResNet-50 fine-tune | 21.3 min | 42 |
| B VGG16 frozen | 19.5 min | 46 |
| C VGG16 fine-tune | 19.9 min | 45 |

These are estimates from 2 timed epochs on a random timing sample, not the project split; no accuracy from this notebook is used. Real training times are logged per run in `experiments/log.csv`.

## Data ethics and limitations

HAM10000 is a public, de-identified research dataset; no private or scraped personal data is used. The images were collected at two sites (Vienna, Austria and Queensland, Australia), so the model may not generalise to other populations, skin types or imaging devices. Results describe performance on this dataset only. The model must not be used to diagnose or reassure any real patient.

---

## Repository structure

```
ham10000-skin-lesion-classification/
├── README.md            problem, results, setup and run instructions
├── requirements.txt     pinned package versions (Colab, Python 3.13)
├── configs/             one file per experiment configuration
├── data/                download script and README (no raw data in git)
├── notebooks/           EDA and results notebooks (they import from src/)
├── src/                 reusable code: data, models, train, evaluate, predict
├── experiments/         experiment log (CSV) and saved curves
├── models/              final saved model and preprocessing objects
├── app/                 demo application
└── report/              report PDF and slides
```

## Setup (Google Colab)

```bash
git clone https://github.com/rohimaahmed101-sys/ham10000-skin-lesion-classification.git
cd ham10000-skin-lesion-classification
pip install -r requirements.txt
```

Add your Kaggle token to **Colab Secrets** as `KAGGLE_API_TOKEN`, then:

```bash
python data/download.py
```

Training, evaluation and demo commands: *added in later phases.*

## How we work

- Nobody pushes directly to `main` after the initial skeleton.
- Each change: create a branch (`feature/<short-name>`) → commit → open a pull request → **a different team member reviews and merges**.
- Every member commits from their own GitHub account.
- Never commit raw data, the Kaggle token, or `kaggle.json`.

