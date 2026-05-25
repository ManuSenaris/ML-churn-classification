# Customer Churn Classification

Binary classification model to predict customer churn in a telecommunications company, with emphasis on business impact analysis and data leakage prevention.  
Developed as part of **Tecnología Digital VI** at Universidad Torcuato Di Tella (UTDT), 2025.

---

## Problem Statement

Given customer data from a telecommunications company, predict whether a customer will churn (stop using the service) in the following month.

**Target variable:** `Churn` (binary: churned / retained)

The dataset presents **class imbalance (~27% churn)**, which makes accuracy alone a misleading metric — a naive model predicting "no churn" for every customer would already achieve 73% accuracy without detecting a single at-risk customer.

---

## Exploratory Data Analysis

Key patterns identified during EDA:

- **Class imbalance:** ~27% of customers churned
- **Tenure:** customers with lower tenure show significantly higher churn rates
- **Monthly charges:** higher charges are associated with increased churn probability
- **Contract type:** contract length strongly influences customer retention

These patterns confirm that churn is not random — it is predictable from observable variables.

---

## Preprocessing Pipeline

### Data Cleaning
- Blank strings and invalid values were identified and converted to `NaN` using `pd.to_numeric()`
- Missing values were imputed using `SimpleImputer` with mean strategy, preserving sample size

### Encoding
- **Categorical variables:** One-Hot Encoding via `get_dummies()` — ordinal encoding was deliberately avoided to prevent introducing artificial ordering between unordered categories
- **Numerical variables:** Standardized with `StandardScaler` to handle magnitude differences, critical for distance-based models

### Data Leakage Prevention
All transformations (imputation, scaling) were fitted **exclusively on the training set** and then applied to the test set. Performing these steps on the full dataset before splitting would cause data leakage — inflating evaluation metrics and undermining real-world generalization.

**Train/Test split:** 80/20

---

## Validation Strategy

- **Stratified K-Fold Cross-Validation** was used instead of simple Holdout to ensure each fold maintains the original class proportion (~27% churn)
- Without stratification, a random fold could contain very few or no examples of the minority class, producing unreliable performance estimates

---

## Handling Class Imbalance

**Oversampling** was applied to address the class imbalance. Undersampling was evaluated but discarded — it would reduce the dataset from 5,634 to 2,990 observations, eliminating valid information and reducing variability in the majority class.

---

## Business Impact Analysis

The choice of evaluation metric depends directly on the business context:

| Scenario | Error to minimize | Metric to maximize |
|---|---|---|
| Discount is costly — avoid unnecessary offers | False Positives | **Precision** |
| Discount is cheap — catch every at-risk customer | False Negatives | **Recall** |

**False Positive:** model predicts churn, but customer was loyal → company offers a 50% discount for 3 months unnecessarily → direct revenue loss.

**False Negative:** model predicts retention, but customer churns → no offer is made → customer leaves → full lifetime value is lost.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | ~0.84 (misleading due to class imbalance) |
| Recall (churn) | 0.45 |
| Precision (churn) | 0.58 |

**Interpretation:**
- The model detects ~45% of customers who actually churned — if the discount retains them, nearly half of at-risk customers are saved
- 55% of churners go undetected and are lost
- 42% of customers offered a discount would not have churned — representing unnecessary cost

The model has moderate effectiveness: it recovers a relevant portion of at-risk customers, but at a considerable rate of unnecessary spending. Whether to deploy it depends on the balance between discount cost and customer lifetime value.

---

## Key Takeaways

- Accuracy is not a sufficient metric in imbalanced classification problems — always evaluate Precision, Recall, and F1-Score alongside the confusion matrix
- Stratified K-Fold is preferred over simple Holdout for imbalanced datasets
- Data leakage is a subtle but critical issue: fit transformers only on training data
- Threshold tuning could improve Recall if minimizing customer loss is the primary objective
- Assigning discounts proportional to predicted churn probability (rather than binary yes/no) could optimize retention costs in future iterations

---

## Technologies

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn (KNN, Logistic Regression, Random Forest), Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## Project Structure

```
├── notebook.ipynb        # Full pipeline: EDA, preprocessing, modeling, evaluation
└── data/
    └── telco_churn.csv   # Telco customer dataset
```
