# CPS 844 Project: Income Prediction Using Machine Learning

**Authors:** Kelvin Chow, Gordon Huang  
**Date:** April 5, 2025

## Overview

This project applies various data mining and machine learning techniques to predict whether an individual's income exceeds $50,000 using the Adult Income dataset from the UCI Machine Learning Repository.

The analysis includes:
- Exploratory data analysis
- Data preprocessing
- Model training and evaluation
- Feature selection using Recursive Feature Elimination (RFE)

We evaluate the performance of five classification algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- Decision Tree
- Random Forest

## Dataset

**Source:** [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)  
**Instances:** 32,561 (original), 24,944 (after cleaning)  
**Target:** Income (`<=50K` or `>50K`)  
**Class Distribution:** 74% `<=50K`, 26% `>50K`

### Features
- Numerical: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`

## Preprocessing Steps

- Dropped irrelevant/redundant features (e.g., `education`)
- Removed rows with missing values
- One-hot encoding for low-cardinality categorical features
- Frequency encoding for high-cardinality features
- Binary encoding for `sex` and `income`
- Standardization for numerical features
- Train-test split using stratified sampling (80/20)

## Model Training

All models were evaluated using stratified 5-fold cross-validation.

### Accuracy with All Features
| Model              | Accuracy  |
|-------------------|-----------|
| Logistic Regression | 0.8337    |
| SVM                 | 0.8360    |
| KNN                 | 0.8131    |
| Decision Tree       | 0.7918    |
| Random Forest       | 0.8382    |

## Feature Selection

Used RFE with Random Forest to select the 7 most important features:

- `age`
- `capital-gain`
- `hours-per-week`
- `education-num`
- `marital-status_Married-civ-spouse`
- `occupation`
- `relationship_Husband`

### Accuracy with Top 7 Features
| Model              | Accuracy  |
|-------------------|-----------|
| Logistic Regression | 0.8286    |
| SVM                 | 0.8353    |
| KNN                 | 0.8188    |
| Decision Tree       | 0.7888    |
| Random Forest       | 0.8195    |

## Classification Metrics (All Features)

**SVM (Best Overall Performer):**
- Accuracy: 0.84
- Precision (<=50K / >50K): 0.86 / 0.74
- Recall (<=50K / >50K): 0.93 / 0.56
- F1-score (<=50K / >50K): 0.89 / 0.64

## Conclusions

- SVM and Random Forest consistently outperformed other models.
- Feature selection marginally affected model performance but did not drastically reduce accuracy.
- Important features align with economic intuition (e.g., age, education, hours worked).
- The study underscores the value of preprocessing, model selection, and feature engineering.

## References

- Dua, D., & Graff, C. (2019). UCI Machine Learning Repository: Adult Data Set. [Link](https://archive.ics.uci.edu/dataset/2/adult)
