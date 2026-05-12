# 📘 Customer Churn Prediction 

This repository contains a complete end-to-end machine learning pipeline for predicting telecom customer churn.  
It includes data preprocessing, EDA, feature engineering, six individually tuned ML models, and performance evaluation using ROC-AUC and confusion matrices.

---

# 🚀 Project Overview

The goal of this project is to predict whether a customer will churn (leave the service) using historical data such as:

- Billing patterns  
- Service usage  
- Payment behavior  
- Demographic information  
- Tenure & account details  

This pipeline follows a production-style workflow suitable for real-world ML applications.

---

# ✨ Key Features

- Full data cleaning & preprocessing  
- Extensive EDA with visual insights  
- Feature engineering + correlation filtering  
- Six tuned ML models (not default settings)  
- ROC curves & confusion matrices for all models  
- AUC-based comparison  
- Clean, silent training (no warnings/log spam)  
- Modern scikit-learn plotting APIs  

---

# 🧹 Data Preprocessing

✔ *Missing Value Handling*
- Numerical → median / mean  
- Categorical → mode or "Unknown"

✔ *Outlier Treatment*
- IQR method  
- Winsorization for extreme skew  

✔ *Encoding*
- Label Encoding (ordinal)  
- One-Hot Encoding (nominal)

✔ *Scaling*
- Only where needed (tree models don’t require scaling)

✔ *Class Imbalance*
- Handled using:  
  - class_weight="balanced"

---

# 📊 Exploratory Data Analysis (EDA)

You performed:

✔ *Target Variable Study*
- Churn distribution  
- Imbalance analysis  

✔ *Univariate Analysis*
- Histograms  
- KDE plots  
- Boxplots  
- Countplots  

✔ *Bivariate Analysis*
- Churn vs tenure  
- Churn vs billing  
- Churn vs services  
- Churn vs payment method  

✔ *Correlation Heatmap*
- Used for removing redundant variables.

✔ *Feature Importance*
- Extracted from tree-based models.

---

# 🧠 Machine Learning Models Used (ALL Tuned)

You trained 6 tuned models — each hyperparameter optimized separately:

- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Machine (GBM - sklearn)  
- XGBoost Classifier  
- LightGBM Classifier  
- CatBoost Classifier  

---

# ⚙ Training Workflow

- *Train Set*: Model training  
- *Validation Set*: Hyperparameter tuning & early stopping  
- *Test Set*: Final evaluation  

Boosting models use:
- Early stopping  
- AUC as evaluation metric  
- Callback-based clean logging  

---

# 📈 Evaluation Metrics

All models were evaluated using:

- ROC-AUC (primary metric)  
- Precision, Recall, F1-score  
- Confusion Matrix  
- Classification Report  
- ROC curves for Test + Validation sets  

### Modern ROC Curve Code

RocCurveDisplay.from_estimator(model, X_test, y_test)



---

# 📘 Dataset Description

✔ *Dataset Contains*
- Demographic information  
- Billing & payment data  
- Tenure & account history  
- Service usage metrics  
- Service quality indicators  
- Churn label (0 or 1)

✔ *Dataset Size*
- Approximately 50,000+ rows  
- Around 50+ processed features after encoding  

✔ *Cleaning Steps Applied*
- Missing value imputation  
- Outlier correction  
- Encoding (Label + One-Hot)  
- Correlation filtering  
- Imbalance handling  

---

# 🏆 Final Model Results

<img width="293" height="292" alt="image" src="https://github.com/user-attachments/assets/41442412-c785-4d15-bed5-9a3ae5d8f09a" />

<img width="281" height="277" alt="image" src="https://github.com/user-attachments/assets/851985d2-1017-4e3a-b8c5-c761cc0bb8a2" />

---

# 📚 Requirements

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- catboost
- joblib
- jupyter

 ---

  # 🔮 Future Improvements

- Deploy with FastAPI
  
- Build a Streamlit dashboard
  
- Use Optuna/Bayesian search for hyperparameter tuning
   
- Automate retraining pipeline

 ---
  

👨‍💻 Developed By :-

K Prudhvi Sai Ram

🏫 IIIT Sri City
