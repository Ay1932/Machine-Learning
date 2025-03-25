# 📊 Customer Churn Prediction ML Project 🚀

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![License](https://img.shields.io/badge/License-MIT-orange)

## 👨🏻‍💻 Project Overview
This project focuses on analyzing customer behavior and predicting **customer churn** using machine learning to help businesses improve retention and reduce churn rates.

### 🔹 Key Highlights
✅ Predicts whether a customer will **churn or stay**  
✅ Identifies **high-risk customers** before they leave  
✅ Helps businesses **take proactive retention steps**  
✅ Improves **customer satisfaction & profitability**  

## 📌 Objectives
✔ Analyze customer behavior and subscription patterns  
✔ Identify key factors causing churn  
✔ Build and compare machine learning models  
✔ Generate actionable business recommendations  

## 📈 Dataset
The dataset contains information about telecom customers:

| Feature | Description |
|---------|------------|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male/Female |
| `Tenure` | Months with company |
| `MonthlyCharges` | Amount paid monthly ($) |
| `TotalCharges` | Total amount paid ($) |
| `Contract` | Month-to-month, 1-year, 2-year |
| `PaymentMethod` | Payment type |
| `Churn` | Target variable (Yes/No) |

📌 **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 📊 EDA Insights

### 📍 Customer Distribution
✅ Most customers are on **month-to-month contracts** (high churn risk)  
✅ **Senior citizens** have higher churn rates  

### 📍 Spending Behavior
✅ Higher monthly charges → More churn  
✅ Longer tenure → Less churn  

### 📍 Key Visualizations
![Churn by Contract](assets/churn_contract_chart.png)  
![Monthly Charges vs Churn](assets/churn_monthly_charges.png)  

## 📋 Project Pipeline
1️⃣ **Data Preprocessing**: Handle missing values, encode features, normalize data  
2️⃣ **Feature Engineering**: Identify key churn factors  
3️⃣ **Model Training**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN  
4️⃣ **Evaluation**: Accuracy, Precision, Recall, F1-score  
5️⃣ **Deployment**: Save best model, generate recommendations  

## ⚖️ Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.7987** | 0.6771 | **0.5791** | **0.6243** |
| Decision Tree | 0.7307 | 0.4933 | 0.4906 | 0.4919 |
| Random Forest | 0.7902 | 0.6579 | 0.4692 | 0.5477 |
| SVM | 0.7941 | **0.6850** | 0.5013 | 0.5789 |
| KNN | 0.7519 | 0.5465 | 0.5040 | 0.5244 |

🏆 **Best Model:** Logistic Regression (Highest accuracy & F1-score)  

## 🛠️ Model Deployment
The best model was saved as `customer_churn_model.pkl` for real-time predictions:

```python
import pickle

def predict_churn(customer_data):
    input_data = np.array([customer_data[feature] for feature in feature_names]).reshape(1, -1)
    prediction = model.predict(input_data)
    return "Churn" if prediction[0] == 1 else "No Churn"
