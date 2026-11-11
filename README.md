# CreditWise: Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved, built with Python and Scikit-learn.

## Dataset
`loan_approval_data.csv`: 1,000 applicants, 19 features (income, credit score, DTI ratio, savings, loan details, etc.).

## Steps
1. Handled missing values (mean for numeric, most frequent for categorical)
2. Exploratory data analysis (histograms, box plots, correlation heatmap)
3. Encoding (Label Encoding and One-Hot Encoding)
4. Train/test split (80/20) and feature scaling (StandardScaler)
5. Trained and compared Logistic Regression, KNN and Naive Bayes
6. Feature engineering (squared Credit Score and DTI ratio)

## Results
Credit Score was the strongest predictor of approval. After feature engineering, Logistic Regression reached **88% accuracy** (up from 86.5%).

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Author
Shivam Padhi: [LinkedIn](https://www.linkedin.com/in/shivam-padhi-77b431331)
