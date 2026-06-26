# 🛢️ Oil Spill Detection using Machine Learning
 
A binary classification system that detects oil spills from satellite-derived numerical features using multiple machine learning models. Handles severe class imbalance through SMOTE oversampling and class weighting.
 
---
 
## 📋 Table of Contents
 
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
---
 
## Problem Statement
 
Oil spills pose a catastrophic threat to marine ecosystems. Early detection using satellite remote sensing is critical for minimising environmental damage. This project frames oil spill detection as a **binary classification** task — given 49 numerical features extracted from a satellite image patch, predict whether an oil spill is present or not.
 
---
 
## Dataset
 
- **Source:** [Kaggle Oil Spill Dataset](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)
- **File:** `oil_spill.csv`
- **Samples:** 937 total
- **Features:** 49 numerical features (`f_1` – `f_49`) — geometric, spectral, and textural properties
- **Target:** Binary (`0` = Non-Oil Spill, `1` = Oil Spill)
| Class | Count | Percentage |
|---|---|---|
| Non-Oil Spill (0) | 896 | 95.6% |
| Oil Spill (1) | 41 | 4.4% |
 
> ⚠️ **Class Imbalance:** The dataset has a 21.9:1 imbalance ratio, addressed using SMOTE and class weighting.
 
---
 
## Methodology
 
### 1. Preprocessing
- Stratified 80/20 train/test split to preserve class proportions
- Feature standardisation with `StandardScaler` (fit on train only)
- **SMOTE** oversampling applied to training set only (749 → 1,432 balanced samples)
### 2. Models Trained
 
| Model | Key Parameters |
|---|---|
| Random Forest | 300 estimators, `class_weight='balanced'` |
| Gradient Boosting | 200 estimators, `lr=0.05`, `max_depth=4` |
| Logistic Regression | `C=1.0`, `class_weight='balanced'`, L2 regularisation |
| MLP Neural Network | Layers: 128→64→32, ReLU, Adam, early stopping |
 
### 3. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- 5-Fold Stratified Cross-Validation
---
 
## Results
 
### Model Comparison (Test Set)
 
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 95.74% | 50.00% | 50.00% | 0.500 | 0.922 |
| **Gradient Boosting** ✅ | **96.28%** | **55.56%** | **62.50%** | **0.588** | **0.925** |
| Logistic Regression | 90.43% | 27.27% | 75.00% | 0.400 | 0.937 |
| MLP Neural Network | 95.21% | 44.44% | 50.00% | 0.471 | 0.918 |
 
> ✅ **Best Model:** Gradient Boosting — highest F1-Score (0.588) and ROC-AUC (0.925)
 
### Confusion Matrix — Gradient Boosting
 
|  | Predicted: Non-Spill | Predicted: Oil Spill |
|---|---|---|
| **Actual: Non-Spill** | 176 ✅ (TN) | 4 ⚠️ (FP) |
| **Actual: Oil Spill** | 3 ❌ (FN) | 5 ✅ (TP) |
 
---
 
## Installation
 
### Prerequisites
- Python 3.8+
- pip
### Clone the repository
 
```bash
git clone https://github.com/priyansh-018/oil-spill-detection.git
cd oil-spill-detection
```
 
### Install dependencies
 
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
 
---
 
## Usage
 
### Run the full pipeline
 
```bash
python oil_spill_classification.py
```
 
This will:
1. Load and explore `oil_spill.csv`
2. Split data into train/test sets
3. Apply SMOTE oversampling
4. Train all 4 models
5. Print metrics table and classification report
6. Save `oil_spill_results.png` (evaluation dashboard)
7. Run 5-fold cross-validation on the best model
### Predict on a new sample
 
```python
from oil_spill_classification import predict_new
 
# Provide a list of 49 feature values
sample = [1, 2558, 1506.09, 456.63, 90, ...]  # f_1 to f_49
result = predict_new(sample)
 
print(result)
# {'label': 'Oil Spill', 'probability': 0.823, 'confidence': 82.3}
```
 
### Adjust decision threshold
 
By default the threshold is `0.5`. Lower it to prioritise recall (catch more spills at the cost of more false alarms):
 
```python
result = predict_new(sample, threshold=0.3)
```
 
---
 
## Model Evaluation
 
The evaluation dashboard (`oil_spill_results.png`) contains six panels:
 
| Panel | Description |
|---|---|
| Metrics Comparison | Bar chart of all metrics across all models |
| Confusion Matrix | TP / TN / FP / FN breakdown for the best model |
| ROC Curves | AUC curves for all models |
| Precision-Recall Curves | Average precision for all models |
| Feature Importances | Top 15 features from Random Forest |
| Class Distribution | Train (post-SMOTE) vs Test class counts |
 
---
 
## Technologies Used
 
| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations and SMOTE implementation |
| `scikit-learn` | Model training, evaluation, cross-validation |
| `matplotlib` | Plotting and visualisation |
| `seaborn` | Confusion matrix heatmap |
 
---
 
## Acknowledgements
 
- Dataset: [Kaggle Oil Spill Detection](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)
- Scikit-learn: Pedregosa et al., JMLR 2011
- SMOTE: Chawla et al., JAIR 2002
