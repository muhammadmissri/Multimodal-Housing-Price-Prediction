Multimodal ML: Housing Price Prediction Using Images + Tabular Data

🎯 Objective
Predict housing sale prices using both structured/tabular data (area,
bedrooms, bathrooms, location score, condition, etc.) and house images,
by:
Extracting visual features from house images using a Convolutional
Neural Network (CNN).
Fusing those CNN-derived features with tabular features.
Training a regression model on the fused representation to predict price.
Evaluating with MAE and RMSE, benchmarked against tabular-only
and image-only baselines to quantify the benefit of multimodal fusion.
🗂️ Repository Structure
```
.
├── notebook/
│   └── Multimodal\_Housing\_Price\_Prediction.ipynb   # main deliverable
├── src/
│   ├── generate\_dataset.py     # builds the synthetic multimodal dataset
│   └── build\_notebook.py       # programmatically assembles the notebook
├── data/
│   ├── housing.csv             # tabular data (generated)
│   └── images/                 # house images, one per row (generated)
├── outputs/                    # saved charts/figures from the notebook run
├── requirements.txt
└── README.md
```
📊 Dataset
This project uses a synthetically generated multimodal dataset
(`src/generate\_dataset.py`) consisting of 600 houses, each with:
Column	Description
`area\_sqft`, `bedrooms`, `bathrooms`, `stories`, `age\_years`, `garage`, `location\_score`, `condition`	Tabular features
`image\_file`	Filename of a matching procedurally generated house image
`price`	Target variable (sale price, USD)
Why synthetic data? A real Kaggle "Housing Prices + Images" dataset
normally backs this task, but it requires internet access to download. The
generator instead builds a synthetic dataset where a shared latent "house
quality" variable drives both the tabular price formula and visual
attributes of the generated image (façade tone, window count, yard
greenery, garage presence, wear/condition speckling) — so the image
modality carries genuine, learnable correlation with price, just like real
listing photos would.
➡️ To use a real dataset: drop in any Kaggle housing CSV + photo folder,
keep an `image\_file` column mapping rows to filenames, and the notebook
runs unmodified.
🧠 Methodology / Approach
Preprocessing
Tabular features standardized with `StandardScaler` (fit on train only).
Images resized to 64×64 and normalized to `\[0, 1]`.
80 / 10 / 10 train / validation / test split.
Models compared
Tabular-only: Random Forest Regressor on structured features.
Image-only: Small CNN (3 conv blocks + global average pooling +
dense head) trained from scratch directly on images.
Multimodal Fusion (core deliverable): the same CNN's image-feature
branch is concatenated with a small dense tabular branch, then
passed through further dense layers to a single price output.
> Note: the CNN is trained from scratch rather than using a pretrained
   > ImageNet backbone (e.g. MobileNetV2/ResNet50) because the development
   > sandbox had no internet access to download pretrained weights. On a
   > machine with internet access, swapping in a frozen pretrained backbone
   > for the image branch is a recommended upgrade for real photo datasets.
Evaluation: MAE, RMSE, and R² computed on a held-out test set for
all three models, plus visualizations (predicted-vs-actual scatter,
residual histogram, training curves, correlation matrix).
✅ Key Results
Model	MAE (USD)	RMSE (USD)	R²
Multimodal Fusion (CNN + Tabular)	~14,200	~18,300	~0.985
Tabular-only (Random Forest)	~17,500	~22,300	~0.977
Image-only (CNN)	~19,000	~22,700	~0.977
(Exact values are produced fresh each run — see the notebook's Section 6
for the live numbers; relative ordering is the meaningful result.)
Observations:
The multimodal fusion model outperformed both single-modality
baselines on every metric, confirming that combining CNN-derived image
features with tabular features adds real predictive value.
The tabular-only model was a strong baseline, since structured features
explain most of the price variance — consistent with real housing data.
The image-only model was weakest alone (pixels can't recover exact
square footage), but it still learned meaningful visual cues, which is
exactly why fusing it with tabular data helped overall.
Takeaway: multimodal fusion pays off most when image and tabular
data carry complementary information rather than fully redundant
information — exactly the design of this experiment.
🛠️ Tech Stack
`TensorFlow / Keras` (CNN + fusion model) · `scikit-learn` (Random Forest
baseline, preprocessing, metrics) · `pandas` / `NumPy` · `Pillow` (image
generation/loading) · `Matplotlib` (visualizations)
▶️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook/Multimodal\_Housing\_Price\_Prediction.ipynb
```
The notebook automatically generates `data/housing.csv` and
`data/images/` on first run if they don't already exist.
📌 Skills Gained
Multimodal machine learning · Convolutional Neural Networks (CNNs) ·
Feature fusion (image + tabular) · Regression modeling and evaluation
(MAE / RMSE / R²)
