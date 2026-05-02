# Forest Cover Type Prediction
### CS-245: Machine Learning | DS-2-A | Group Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)

---

## Project Overview

This project predicts the **forest cover type** of a 30x30 meter land cell in Roosevelt National Forest, Colorado using cartographic measurements. It is a multi-class classification problem with **7 forest cover types**.

We implemented and compared:
- Two supervised learning models (Logistic Regression + Random Forest)
- One unsupervised learning model (K-Means Clustering)
- Bonus: Interactive Streamlit web application

---

## Group Members

| Name | CMS ID |
|------|--------|
| Ayesha Mobeen | 513958 |
| Sara Fawad | 509615 |
| Iqra Nisar | 501191 |
| Aneeqa Zahid | 519265 |

---

## Repository Structure

```
Forest-Cover-Type-ML/
│
├── ML_project.ipynb          # Main Jupyter Notebook (full pipeline)
├── forest_app.py             # Streamlit web application (bonus)
├── Report.docx               # Project report
└── README.md                 # This file
```

---

## Dataset

**Name:** UCI Forest Cover Type Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/31/covertype)  
**Size:** 581,012 instances | 54 features | 7 classes  
**Missing Values:** None

> The dataset (covtype.csv) is not included in this repository due to file size limits.  
> Please download it from the link above and place it in the project folder before running.

### Cover Types

| Class | Forest Type |
|-------|------------|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## Models Implemented

### 1. Logistic Regression (Baseline)
- Multinomial classification with L2 regularization
- class_weight='balanced' to handle class imbalance
- solver='lbfgs', max_iter=1000

### 2. Random Forest (Best Model)
- Ensemble of 200 decision trees
- Handles non-linear boundaries natively
- Superior performance on minority classes

### 3. K-Means Clustering (Unsupervised)
- Applied to 10 continuous features
- PCA used to reduce to 2D for visualization
- Silhouette score used for evaluation

---

## Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | ~71% | ~0.68 |
| Random Forest | ~94% | ~0.93 |
| K-Means (Silhouette Score) | — | ~0.18 |

Random Forest significantly outperforms Logistic Regression, especially on minority classes such as Cottonwood/Willow and Aspen.

---

## How to Run

### Run the Jupyter Notebook
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
jupyter notebook ML_project.ipynb
```

### Run the Streamlit App (Bonus)
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
streamlit run forest_app.py
```

Then open your browser at http://localhost:8501

App workflow: Upload covtype.csv in the sidebar, click Train All Models Now, then go to the Predict tab.

---

## Streamlit App Features

| Tab | Description |
|-----|-------------|
| Predict | Enter cartographic values and get a cover type prediction |
| Train Models | Upload dataset and train all 3 models in the browser |
| Compare Models | Metrics, confusion matrices, and feature importance |
| Clustering | K-Means PCA visualization |
| About | Project and model details |

---

## Tech Stack

- Python 3.x
- scikit-learn — ML models and evaluation
- pandas and numpy — Data processing
- matplotlib and seaborn — Visualizations
- Streamlit — Web application

---

## Contact

GitHub: [iqranisar04](https://github.com/iqranisar04)
