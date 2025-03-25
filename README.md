# Machine-Learning
Machine learning with Supervised Learning
🛍️ E-Commerce Customer Behaviour Analysis ML Project
👨🏻‍💻 Project Overview
This project observed e-commerce customer behaviour in online shopping throughout my machine learning knowledge and techniques. Using a dataset of e-commerce customer behaviour, it obseved important aspects that influence online shopping and developing online shopping prediction models to enhance our business.

📌 Objectives
Analyze customer behaviour, purchase habit and satisfaction level.
Identify how customers attracted, and how enhance business opportunity.
Observation of different value comparison so that we focuses on our actual customers.
Focus on our targeted audience, and improve our sales.
It Helps businesses archive their goals and optimize their valuable assets.
📈 Dataset
The dataset contains information about E-Commerce Customer Behaviour Dataset including:

Customer ID - Integer
Gender - String
Age - Integer
City - String
Membership Type - String
Total Spend - Numeric
Items Purchased - Integer
Average Rating - Numeric
Discount Applied - Boolean
Days Since Last Purchase - Integer
Satisfaction Level - String
Source: Kaggle - E-commerce Customer Behavior Dataset

📊 Exploratory Data Analysis (EDA) Insights
Key insights extracted from the dataset:

Demographics:
The average customer age is 33.60 years with a broad age distribution.
Spending Behavior:
Customers spend an average of $845.38 per transaction.
Average number of items purchased per order: 12.6 items.
Engagement Levels:
Satisfaction Level Distribution:
Satisfied: 36.30%
Neutral: 30.66%
Unsatisfied: 33.24%
Average days between purchases: 26.59 days.
Correlations:
Strong relationships between customer engagement, total spend, and purchase frequency.
Customers with higher satisfaction tend to spend $595.14.
📋 Project Pipeline
1️⃣ Data Preprocessing: delete recode of missing values, encode category features, and normalize numerical data.
2️⃣ Feature Engineering: Identify key features influencing customer satisfaction and spending.
3️⃣ Model Selection & Training: Training this models like Random Forest, XGBoost, and Logistic Regression.
4️⃣ Evaluation: Use accuracy, precision, recall, RMSE, and F1-score for performance observation.
5️⃣ Deployment & Business Insights: Generate recommendations for customer attension and marketing strategies and tequniques.

⚖️ Machine Learning Models Evaluated
Model	Accuraciy (%)	F1 Score (%)	Training Time (s)
Random Forest	100.0%	100.0%	0.197
SVM	100.0%	100.0%	0.008
Gradient Boosting	100.0%	100.0%	0.324
KNN	98.6%	98.6%	0.003
Naive Bayes	100.0%	100.0%	0.004
Logistic Regression	98.6%	98.6%	0.372
🗝️ Key Insights
The our average customer age is 33.60 years.
Customers spending on an average of $845.38 per transaction.
Average number of items purchased in per order is: 12.6 items.
Satisfaction Level Distribution:
✅ Satisfied: 36.10%
⚠️ Neutral: 30.66%
❌ Un-satisfied: 33.24%
The average days between purchases is 26.59 days.
Higher satisfied consumers spending $1273.50 on an average, while unsatisfied consumers spend only $595.14 on an average.
🗣 Business Recommendations
1️⃣ Boost Retention – Offere 5-10% discounts or loyality to Neutral/Very Unsatisfied consumers to increase their repeat product purchases.
2️⃣ Personalized Marketing – Identifying top consumers and provide VIP services & targeted promotions to enhance our revenue.
3️⃣ Optimize Discounts – give personalized discounting for more product purchase, especially for unsestified consumers.
4️⃣ Enhance Engagement – apply email marketing campign for that customers which is inactive for 30+ days to connect them.

🤖 Technologies Used
Python 3.13
Pandas & NumPy
Plotly & Matplotlib
Scikit-learn
Jupyter Notebook
🚀 Getting Started
Clone this repository
git clone https://github.com/yatikanghan/Machine-Learning.git
Install dependencies
pip install -r requirements.txt
Open the Jupyter notebook
jupyter notebook jupyter/project.ipynb
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
