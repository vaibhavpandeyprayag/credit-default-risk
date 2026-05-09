# Credit Default Risk Classification (German Credit Dataset)

## Overview

This project builds a complete machine learning pipeline to predict credit default risk using the German Credit dataset. The goal is to identify high-risk customers (defaulters) while balancing model performance and generalization.

The project focuses on:

- Handling class imbalance
- Controlling overfitting
- Comparing multiple models
- Evaluating using business-relevant metrics

---

## Problem Statement

Predict whether a customer is likely to default on credit based on financial and demographic attributes.

- Target Variable: `risk`
    - `0` → Good (Non-defaulter)
    - `1` → Bad (Defaulter)

---

## Dataset

- Source: UCI German Credit Dataset
- Samples: 1000
- Type: Structured tabular data
- Link: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

---

## Tech Stack

### Core

- Python 3.9
- NumPy
- pandas

### Visualization

- matplotlib
- seaborn

### Machine Learning

- scikit-learn
- XGBoost
- imbalanced-learn

### Utilities

- joblib (model saving)

---

## Project Structure

```text
credit-risk/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/                # Data loading & cleaning
│   ├── features/            # Feature engineering
│   ├── preprocessing/       # Encoding & scaling
│   ├── pipelines/           # Model pipelines
│   ├── models/              # Evaluation utilities
│   └── config.py
│
├── reports/                 # Metrics, plots, outputs
├── notebooks/               # EDA notebooks
├── main.py                  # Entry point
├── environment.yaml
└── README.md
```

---

## Pipeline

1. **Data Loading**
2. **Data Processing**
    - Missing value handling
    - Target mapping

3. **Feature Engineering**
    - Derived features (e.g., credit per duration)

4. **Train-Test Split**
5. **Model Pipelines**
    - Encoding (OneHot + Ordinal)
    - Scaling (for linear models only)

6. **Model Training**
    - Logistic Regression
    - Random Forest
    - XGBoost

7. **Evaluation**
    - Accuracy, Precision, Recall, F1-score
    - ROC-AUC
    - Confusion Matrix
    - ROC Curve & Precision-Recall Curve

8. **Reporting**
    - Metrics saved as JSON
    - Model comparison saved as CSV
    - Plots saved in `reports/`

---

## Models Used

### Logistic Regression

- Strong baseline
- Good generalization
- Higher recall (detects defaulters better)

### Random Forest (Best Model)

- Balanced precision & recall
- Controlled overfitting
- Best trade-off for business use

### XGBoost

- Strong performance
- Slight overfitting observed
- High potential with further tuning

---

## Key Techniques

- Class imbalance handling:
    - `class_weight`
    - `scale_pos_weight`

- Overfitting control:
    - Tree depth limitation
    - Regularization

- Threshold tuning (for recall improvement)
- Pipeline-based preprocessing (no data leakage)

---

## Results Summary

| Model         | Accuracy | Precision | Recall | F1    | AUC   |
| ------------- | -------- | --------- | ------ | ----- | ----- |
| Logistic      | ~0.66    | ~0.45     | ~0.64  | ~0.53 | ~0.73 |
| Random Forest | ~0.75    | ~0.57     | ~0.66  | ~0.61 | ~0.79 |
| XGBoost       | ~0.76    | ~0.60     | ~0.62  | ~0.61 | ~0.79 |

### Final Conclusion

Random Forest provides the best balance between recall and precision with minimal overfitting, making it the most suitable model for credit risk prediction.

---

## How to Run

### 1. Create Environment

```bash
conda env create -f environment.yaml
```

### 2. Activate Environment

```bash
conda activate credit-risk
```

### 3. Run Project

```bash
python main.py
```

---

## Key Learnings

- Accuracy is not reliable for imbalanced datasets
- Recall is critical in risk-sensitive problems
- Overfitting must be actively controlled
- Tree-based models require regularization
- Threshold tuning can significantly improve performance

---

## Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna)
- SMOTE integration and comparison
- Cross-validation-based evaluation
- Model deployment (FastAPI / Streamlit)
- Experiment tracking (MLflow)

---

## Author

Vaibhav Pandey
