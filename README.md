# ml-freddie
AI/ML models for Loan Risk Assessment
[View the project documentation](./240924_ML_LoanPredictor_FreddieMacReport.pdf)

# Machine Learning Models for Loan Classification Using Real World Data

## Author:
Natalya Sheremetyeva  
Email: [natalya.sheremetyeva@gmail.com](mailto:natalya.sheremetyeva@gmail.com)  

---

## Abstract

In this project, I developed a machine learning model for loan performance classification using XGBoost trained on real-world data from Freddie Mac’s Single-Family Loan-Level dataset. Key features, such as Current Loan Delinquency Status (CLDS) and the engineered Estimated Loan-to-Value (ELTV) feature, emerged as important predictors for the model, as determined by SHAP values. The model achieved an average ROC AUC score of 0.94 on the unseen test set, demonstrating strong predictive ability. Class imbalance was addressed using undersampling techniques, improving predictions for non-performing loans. While the model demonstrated robust performance with an 82.9% profit advantage over a random baseline from 2014-2017, performance declined slightly in later years, highlighting the need for periodic retraining to maintain accuracy in evolving market conditions.

---

## Introduction

Lenders need to manage risk and allocate capital efficiently to maintain financial stability. Loan classification, a crucial part of credit risk assessment, involves predicting whether a loan will perform well or default. This helps lenders make informed decisions about loan approvals, interest rates, and credit limits.

Loan classification models also help lenders meet regulatory requirements and reduce financial losses. In the current economic environment, with fluctuating interest rates, inflation concerns, and economic uncertainty, accurate loan classification has become critical. Recent events like the collapse of Silicon Valley Bank and the buyout of First Republic Bank by JPMorgan emphasize the importance of predictive models for safeguarding the stability of financial institutions and the broader economy.

The objectives of this project are:
1. To train a machine learning classifier that predicts loan performance.
2. To optimize the classifier through hyperparameter tuning and model selection techniques.
3. To identify key features influencing loan outcomes, providing insights into loan performance.

The report is structured as follows: I first discuss the design of the target variable and its connection to loan performance. Then, key model performance metrics are presented, followed by an analysis of feature importance and model stability. The technical details of the model development process are in Section 2.

---
