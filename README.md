# 📊 Customer Churn Prediction Using Machine Learning 🚀  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-yellow)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)  
![GitHub](https://img.shields.io/badge/GitHub-Public-orange)  

## 🌟 Project Overview  
Customer churn is a **critical issue** for businesses, leading to **lost revenue and higher customer acquisition costs**. This project aims to **predict customer churn** using **machine learning models** based on customer data from a telecom company.  

🔹 **Objective:** Predict whether a customer will churn based on subscription details.  
🔹 **Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) *(7,043 customers, 21 features)*  
🔹 **Tech Stack:** `Python`, `Scikit-Learn`, `Pandas`, `Seaborn`, `Plotly`, `Flask` *(Optional for Deployment)*  
🔹 **Models Used:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN  
🔹 **Deployment:** Model saved as a `.pkl` file for real-time predictions.  

---

## 🔍 Exploratory Data Analysis (EDA)  
EDA helped us uncover key insights before training models:  

📌 **Key Findings:**  
✅ Customers with **month-to-month contracts churn the most**.  
✅ **Higher Monthly Charges lead to more churn**.  
✅ **Electronic check users have the highest churn rate**.  

📊 **Churn Rate by Contract Type:**  
![Churn Contract Chart](assets/churn_contract_chart.png)  

📊 **Monthly Charges vs. Churn:**  
![Churn Monthly Charges](assets/churn_monthly_charges.png)  

*(Upload images inside an `assets/` folder and update the image paths above.)*  

---

## ⚙️ Data Preprocessing  
- **Handled Missing Values:** Filled missing values in `TotalCharges`.  
- **Encoded Categorical Variables:** Converted text data to numeric format.  
- **Feature Scaling:** Used `StandardScaler` for models like SVM & KNN.  
- **Handled Imbalanced Data:** Applied `SMOTE` to balance churn vs. non-churn cases.  

---

## 🚀 Model Training & Evaluation  
We trained **five models** and compared their accuracy, precision, recall, and F1-score:  

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|------------|--------|----------|
| **Logistic Regression** | **0.7987** | 0.6771 | **0.5791** | **0.6243** |
| **Decision Tree**       | 0.7307 | 0.4933 | 0.4906 | 0.4919 |
| **Random Forest**       | 0.7902 | 0.6579 | 0.4692 | 0.5477 |
| **SVM**                | 0.7941 | **0.6850** | 0.5013 | 0.5789 |
| **KNN**                | 0.7519 | 0.5465 | 0.5040 | 0.5244 |

✅ **Best Model:** **Logistic Regression** (Highest accuracy & best F1-score)  
✅ **Improved Performance with Hyperparameter Tuning**  

---

## 🛠️ Saving & Deploying the Model  
The best model was **saved as a `.pkl` file** for real-time predictions.  

```python
import pickle

# Save the trained model
model_data = {"model": best_logreg, "feature_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
