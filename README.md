# Child Mind Institute (CMI) - Problematic Internet Use Analysis

Academic project for the Data Mining II course, Master of Science in Data Science and Business Informatics, University of Pisa (Academic Year 2025/2026).

## Project Overview

This project investigates the prediction and behavioral profiling of Problematic Internet Use (PIU) among children and adolescents, utilizing multimodal data provided by the Child Mind Institute (CMI). 

The primary target of interest is the Severity Impairment Index (sii), derived from the Parent-Child Internet Addiction Test (PCIAT). The problem is formulated in two ways:
- Multiclass classification: predicting severity levels across four tiers (0 = None, 1 = Mild, 2 = Moderate, 3 = Severe).
- Binary classification: separating non-problematic (sii = 0) from problematic cases (sii in 1, 2, 3), represented as sii_binary.

The analysis is conducted across two separate data modalities:
1. Tabular clinical and demographic data: 8,460 participants described by 82 features covering demographics, physical measurements, bio-electric impedance analysis (BIA), treadmill endurance tests, physical activity questionnaires (PAQ), and sleep disturbance scales (SDS).
2. Multivariate time-series data: continuous 30-minute interval wrist-worn accelerometer recordings across 200 time steps, measuring spatial acceleration (X, Y, Z), physical motion intensity (enmo), arm inclination (anglez), ambient light, and sensor compliance flags.

## Core Notebooks and Pipeline

The repository is organized around the primary analytical notebooks corresponding directly to the chapters of the project report.

### Part 1: Tabular Data Analysis

- Exploratory data analysis of clinical and behavioral metrics. Analyzes the distribution of sii across sex, age groups, and explores the relationship between daily computer/internet usage hours and sleep disturbances through violin plots.

- Data cleaning, missing value imputation, consensus-based outlier detection utilizing four concurrent algorithms (identifying and removing 93 anomalous records, representing 1.12% of the dataset), and low-dimensional manifold visualizations using PCA, t-SNE, and UMAP.

- Systematic evaluation of class imbalance mitigation techniques on decision tree baselines. Compares undersampling (Condensed Nearest Neighbour), oversampling (SMOTE and ADASYN), and algorithmic cost-sensitive weighting (Class Weight).

- Extensive benchmarking, hyperparameter tuning with cross-validation, and error analysis across multiple families of classifiers: Support Vector Machines (SVM), Random Forest, Bagging, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost.

- Multinomial Logistic Regression: Implementation and evaluation of a multinomial logistic regression model as a parametric linear baseline for the multiclass classification task (sii target). The model is configured with balanced class weighting to address class imbalance and serves as a direct benchmark against non-linear architectures.

- Deep Neural Networks (DNN) with Optuna Tuning: Design, training, and benchmarking of fully connected deep neural networks under four distinct regularization strategies: unregularized baseline (NOREG), L1 Lasso regularization for feature sparsity, L2 Ridge regularization for weight shrinkage, and Dropout regularization. Hyperparameters such as layer depth, hidden unit capacity, learning rates, and regularization coefficients are systematically optimized through Bayesian optimization using Optuna.

- Explainable AI (XAI) with SHAP and LIME: Post-hoc model interpretability applied to
the best-performing neural network model (L2 regularized architecture): Global Explainability (SHAP): calculation of Shapley Additive exPlanations using a representative 100-sample k-means background dataset from the training partition, generating global summary bar plots to quantify feature contributions across all four severity classes.
Local Explainability (LIME): application of LimeTabularExplainer on individual test instances (both correctly classified and misclassified cases) to explain the local decision logic and feature importance that led to specific predictions.

- Formulation of a multiple regression task aimed at predicting the continuous raw sleep disturbance score (SDS-SDS_Total_Raw). Models tuned and benchmarked include LightGBM Regressor, Support Vector Regression (SVR), and Bagging Regressor.

### Part 2: Time Series Analysis

- Exploratory data analysis of accelerometer streams, sensor non-wear flag compliance, activity profiles of the most monitored participants, Hampel filter spike detection and imputation, moving average smoothing, logarithmic scaling of light intensity, dataset-wise z-score normalization, and dimensional reduction via PAA and SAX.

- Matrix Profile calculation (window size m = 8 steps, corresponding to 4 hours), motif discovery, and medoid profile extraction across activity clusters. Discriminative subsequence mining via Random Dilated Shapelet Transform (RDST) coupled with Random Forest, identifying the top behavioral discriminative pattern (Shapelet #701) and bridging motif analysis with shapelet interpretation.

- Sequence clustering using Time Series K-Means with Euclidean and Dynamic Time Warping (DTW) metrics, as well as Hierarchical clustering. Visual validation via 2D projections and centroid inspection. Addresses patient-level bias through a Monte Carlo consensus clustering framework for subjects with multiple recorded sequences. Feature-based clustering evaluation using Catch22 extracted properties.

- Sequence-level supervised classification using non-parametric K-Nearest Neighbors (KNN) under both Euclidean and DTW distance alignments, compared against the MiniRocket convolutional feature transform.

## Key Findings and Experimental Results

- Tabular Modality:
Class imbalance and overlapping questionnaire distributions present significant classification challenges, particularly for intermediate severity grades. Gradient boosting ensembles (LightGBM and XGBoost) and regularized neural networks achieved the most reliable generalization. Explainability via SHAP revealed that daily screen time, physical performance thresholds, and sleep indices serve as the strongest predictors for higher impairment categories. In the regression task on sleep scores, Bagging and LightGBM achieved the best fit with R2 values around 0.11 and Mean Absolute Error around 6.5.

- Time Series Modality:
The Shapelet-based approach (RDST combined with Random Forest) outperformed all other temporal classifiers, attaining a Macro F1-score of 0.62 and a ROC AUC of 0.67 on unseen test data, surpassing KNN DTW (F1 0.56), MiniRocket (F1 0.56), and KNN Euclidean (F1 0.53). Centroid and motif analyses indicated that behavioral differentiation is primarily driven by sustained postural states (arm inclination angle and specific spatial resting positions) rather than overall physical activity volume alone.

## Requirements and Installation

The environment requires Python 3.10 or higher. The main dependencies include:

- Data manipulation and numerics: numpy, pandas, scipy
- Machine learning and sampling: scikit-learn, imbalanced-learn
- Boosting frameworks: lightgbm, xgboost, catboost
- Deep learning: tensorflow, keras
- Hyperparameter optimization: optuna
- Model interpretability: shap, lime
- Time series processing and modeling: sktime, aeon, stumpy, matrixprofile-ts, pycatch22
- Visualization: matplotlib, seaborn