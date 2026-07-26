<p align="center">
  <img src="assets/banner.png" alt="Credit Card Default Prediction" width="100%" height="180">
</p>

<h1 align="center">Credit Card Default Prediction</h1>

<p align="center">
  <strong>Predicting credit card payment defaults using classical ML models</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#about">About</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#models--results">Models & Results</a> •
  <a href="#live-demo">Live Demo</a> •
  <a href="#project-structure">Project Structure</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/streamlit-1.60-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## Quick Start

```bash
# clone the repo
git clone https://github.com/buildwithsushmita/credit-card-default-prediction.git
cd credit-card-default-prediction

# install dependencies
pip install -r requirements.txt

# train all models (downloads dataset automatically)
cd model && python train_models.py && cd ..

# launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## About

Can we predict whether a credit card holder will default on their next payment? This project tackles that question by training and comparing 5 classical machine learning classifiers on real-world credit data from Taiwan.

The entire pipeline, from data ingestion to an interactive web dashboard, is built as a single deployable application using Streamlit.

**Key highlights:**
- 30,000 real client records with 23 features
- 5 models trained and benchmarked head-to-head
- 6 evaluation metrics per model (Accuracy, AUC, Precision, Recall, F1, MCC)
- Interactive web app for exploring predictions and model performance

---

## Dataset

**Default of Credit Card Clients** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

| Property | Value |
|----------|-------|
| Instances | 30,000 |
| Features | 23 |
| Target | Binary (default = 1, no default = 0) |
| Class split | 77.9% / 22.1% (imbalanced) |
| Origin | Taiwan, 2005 |

Features span credit limit, demographics (age, gender, education, marital status), 6 months of repayment history, bill statement amounts, and previous payment amounts.

---

## Models & Results

<p align="center">
  <img src="assets/model_comparison.png" alt="Model Comparison" width="85%">
</p>

### Performance Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.8102 | 0.7213 | 0.6250 | 0.3542 | 0.4521 | 0.3683 |
| KNN | 0.7928 | 0.7016 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes | 0.4160 | 0.6516 | 0.2496 | 0.8176 | 0.3824 | 0.1111 |
| **Random Forest** | **0.8153** | **0.7678** | **0.6558** | 0.3474 | **0.4542** | **0.3815** |

### Observations

| Model | Takeaway |
|-------|----------|
| Logistic Regression | High precision (69%) but very low recall (24%). Conservative, rarely predicts default. Solid baseline. |
| Decision Tree | Better recall (35%) with good interpretability. Depth limit prevents overfitting. |
| KNN | Struggles with 23-dimensional space. More false positives than tree-based methods. |
| Naive Bayes | Best recall (82%) but tanks accuracy (42%). Independence assumption breaks down on correlated payment features. |
| **Random Forest** | **Winner.** Best accuracy, AUC, F1, and MCC. Ensemble captures complex feature interactions that single models miss. |

---

## Live Demo

| Link | Status |
|------|--------|
| [Streamlit App](https://YOUR_APP_URL.streamlit.app) | Deployed |
| [GitHub Repo](https://github.com/buildwithsushmita/credit-card-default-prediction) | Public |

**App features:**
- Upload custom test data (CSV)
- Select any of the 5 models from a dropdown
- View all 6 evaluation metrics at a glance
- Confusion matrix heatmap
- Full classification report
- Side-by-side comparison of all models

---

## Project Structure

```
ML_Assignment2/
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
├── README.md
├── test_data.csv          # Held-out test split (6,000 rows)
├── model_results.csv      # Pre-computed metrics
├── assets/                # Images for README
│   ├── hero_banner.png
│   └── model_comparison.png
└── model/
    ├── train_models.py    # Training pipeline
    ├── scaler.pkl         # StandardScaler
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| scikit-learn | Model training and evaluation |
| Streamlit | Interactive web dashboard |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | Visualizations |
| joblib | Model serialization |

---

<p align="center">
  <sub>Built as part of the Machine Learning course, M.Tech (AI ML), BITS Pilani - Work Integrated Learning Programmes.</sub>
</p>
