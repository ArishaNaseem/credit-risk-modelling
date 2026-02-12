# Credit Risk Prediction Using XGBoost

## 📌 Project Overview

This project develops a machine learning system to classify loan applicants as:

* **Good Risk (Low Probability of Default)** → `1`
* **Bad Risk (High Probability of Default)** → `0`

The solution follows a complete machine learning pipeline:

**Data Analysis → Feature Engineering → Model Training → Evaluation → Web Deployment**

The final model is deployed as an interactive **Streamlit web application** for real-time credit risk assessment.

---

## 📊 Dataset

The model is trained on the **German Credit Dataset**, which includes financial and demographic attributes such as:

* Age
* Sex
* Job Category
* Housing Type
* Saving Accounts Status
* Checking Account Status
* Credit Amount
* Loan Duration
* Purpose of Loan

These variables are used to predict whether a borrower is likely to default.

---

## 🔎 Exploratory Data Analysis (EDA)

Performed detailed analysis to understand feature influence on credit risk:

* Distribution analysis of Age, Credit Amount, and Duration
* Boxplots for spread and anomaly detection
* Countplots for categorical variable behavior
* Correlation heatmap of financial variables
* Risk-wise comparisons of numerical features
* Scatter and violin plots to observe interaction effects

**Key Observations:**

* Larger credit amounts and longer durations increase default likelihood.
* Account status strongly correlates with repayment behavior.

---

## 🛠️ Data Preprocessing & Feature Engineering

### Missing Value Treatment

Rows with missing categorical information were removed to maintain clean inputs.

```python
df = df.dropna().reset_index(drop=True)
```

### Feature Selection

Final features used for modelling:

```
["Sex","Age","Job","Housing",
 "Credit amount","Saving accounts",
 "Checking account","Duration"]
```

### Encoding Categorical Variables

Categorical variables were transformed using **Label Encoding**, and encoders were saved using `joblib` to ensure identical transformation during deployment.

```python
joblib.dump(le, f"{col}_encoder.pkl")
```

### Train-Test Split

Used stratified sampling to preserve class distribution.

---

## 🤖 Model Development – XGBoost Classifier

The final model uses **Extreme Gradient Boosting (XGBoost)**, chosen for its ability to:

* Capture nonlinear relationships
* Handle structured/tabular data effectively
* Improve predictive power through boosting
* Provide strong generalization on small-to-medium datasets

Hyperparameters were optimized using **GridSearchCV (5-fold Cross-Validation)**.

---

## ⚖️ Handling Class Imbalance

Since credit datasets contain uneven risk distribution, class imbalance was handled using:

```python
scale_pos_weight = (Number of Bad Cases) / (Number of Good Cases)
```

This ensures the model focuses on identifying risky applicants.

---

## 📈 Model Evaluation

Performance was evaluated using multiple classification metrics instead of relying only on accuracy.

| Metric         | Value     |
| -------------- | --------- |
| Accuracy       | **64.7%** |
| ROC–AUC        | **0.707** |
| Macro F1-score | **0.64**  |

### Classification Performance

| Class         | Precision | Recall | F1-score |
| ------------- | --------- | ------ | -------- |
| Bad Risk (0)  | 0.59      | 0.63   | 0.61     |
| Good Risk (1) | 0.70      | 0.66   | 0.68     |

### Confusion Matrix

```
[[29 17]
 [20 39]]
```

**Interpretation:**

* Correctly identified 63% of risky applicants.
* Maintained balanced detection across both safe and risky customers.
* ROC–AUC ≈ 0.71 indicates meaningful discrimination capability.

---

## 💾 Model Persistence

The trained model and encoders were serialized using `joblib`:

```python
joblib.dump(best_xgb, "xgb_credit_model.pkl")
```

This allows reuse without retraining.

---

## 🌐 Streamlit Web Application

An interactive user interface was built using **Streamlit** to enable real-time predictions.

### Features:

* User inputs applicant information
* Automatic encoding using saved transformers
* Instant Good/Bad Risk classification
* Lightweight and deployable interface

### Run Locally:

```
streamlit run app.py
```

---

## 🧰 Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

---

## 📂 Project Structure

```
credit-risk-prediction/
│
├── app.py
├── analysis_model.ipynb
├── german_credit_data.csv
├── xgb_credit_model.pkl
├── *_encoder.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Key Learnings

* Built an end-to-end ML pipeline for financial risk prediction
* Applied feature encoding and reproducible preprocessing
* Tuned gradient boosting models using cross-validation
* Evaluated models using ROC-AUC and F1-score
* Deployed ML model into an interactive web application


**Arisha Naseem**
Engineering Student | Machine Learning & Data Analytics
