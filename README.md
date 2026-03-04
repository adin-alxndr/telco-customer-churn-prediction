# Telco Customer Churn Prediction

## 📌 Project Overview
This project aims to build a machine learning model to predict customer churn in a telecommunications company.  
Customer churn occurs when customers stop using a company's services, which can significantly impact revenue.

By identifying customers who are likely to churn, the company can take proactive actions to retain them.

---

## 🎯 Objective
- Predict whether a customer will churn or not based on customer attributes.
- Identify the most important factors that influence churn.
- Provide business recommendations to reduce customer churn.

---

## 📊 Dataset
**Source:** Kaggle – Telco Customer Churn  
**Rows:** 7,043  
**Columns:** 21  
**Target Variable:** `Churn` (Yes = churn, No = not churn)

Each row represents a customer, and each column represents customer attributes such as:
- Demographics (gender, senior citizen, partner, dependents)
- Services (internet, phone, streaming, security)
- Account information (tenure, contract, payment method, charges)

---

## 🧠 Methodology
1. Data Loading (directly from Kaggle)
2. Data Cleaning  
   - Convert `TotalCharges` to numeric  
   - Handle missing values  
   - Drop irrelevant column (`customerID`)
3. Exploratory Data Analysis (EDA)  
   - Churn distribution  
   - Tenure and monthly charge analysis  
4. Feature Engineering  
   - Encode categorical variables  
5. Model Training  
   - Logistic Regression  
   - Random Forest Classifier  
6. Model Evaluation  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - ROC-AUC  
7. Feature Importance Analysis  
8. Business Insight & Recommendation  

---

## ⚙️ Tools & Libraries
- **Python**
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  
- Kaggle API  
- VSCode 
- GitHub  

---

## 📈 Model Performance

### Logistic Regression
- Accuracy: **0.80**
- Recall (Churn): **0.55**

### Random Forest
- Accuracy: **0.79**
- Recall (Churn): **0.49**

**Best model:** Logistic Regression  
It provides better performance in identifying churn customers and is more interpretable.

---

## 🔍 Key Features Influencing Churn
Top contributing features:
- TotalCharges  
- MonthlyCharges  
- Tenure  
- Contract type  
- PaymentMethod  
- OnlineSecurity  
- TechSupport  

---

## 💡 Business Insights
Customers who are more likely to churn tend to:
- Have short tenure (new customers)
- Pay higher monthly charges
- Use month-to-month contracts
- Not use OnlineSecurity or TechSupport services

---

## 📝 Business Recommendations
- Encourage customers with month-to-month contracts to switch to long-term contracts.
- Offer discounts or bundled packages for customers with high monthly charges.
- Promote OnlineSecurity and TechSupport services as part of retention strategies.
- Focus retention efforts on customers with tenure below 6 months.


---

## 🚀 Future Improvements
- Handle class imbalance using SMOTE or class weighting.
- Perform hyperparameter tuning.
- Try advanced models such as XGBoost or LightGBM.
- Deploy model using Streamlit or FastAPI.

---

## 👤 Author
Name: adin_alxndr

---

## 📎 License
This project is for educational and portfolio purposes only.
