# Titanic Survival Prediction — End-to-End Machine Learning Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/titanic-survival-prediction/blob/main/notebooks/titanic_survival_prediction.ipynb)

A structured binary classification project built in Python and `scikit-learn` to predict passenger survival on the Titanic using demographic, ticket class, and family-size features.

## Project Overview

The goal of this project is to build a reproducible machine learning workflow from raw data inspection to inference on unseen passenger records while preventing data leakage through `Pipeline` and `ColumnTransformer`.

### Key Highlights
- **Exploratory Data Analysis (EDA):** Examined survival patterns across passenger gender (`Sex`), ticket class (`Pclass`), and age distributions.
- **Feature Engineering:** Extracted passenger honorific titles (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`) from `Name` and engineered `FamilySize` and `IsAlone` features.
- **Leakage-Safe Preprocessing:** Applied median imputation + standard scaling to numerical features and most-frequent imputation + one-hot encoding to categorical features inside a `ColumnTransformer`.
- **Model Benchmarking:** Compared **Logistic Regression**, **Random Forest**, and **Gradient Boosting** using 5-fold stratified cross-validation and held-out test metrics (Accuracy and ROC-AUC).
- **Inference Utility:** Included a reusable `predict_passenger_survival()` helper for single-record predictions and survival probability estimates.

## Dataset

- **Source:** [Titanic Dataset (`titanic.csv`)](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Target Variable:** `Survived` (`0` = Did Not Survive, `1` = Survived)
- **Features Used:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, plus engineered `Title`, `FamilySize`, and `IsAlone`.

## Repository Structure

```text
titanic-survival-prediction/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── titanic_survival_prediction.ipynb
