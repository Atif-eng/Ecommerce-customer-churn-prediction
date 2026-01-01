# E-commerce Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Library](https://img.shields.io/badge/Library-XGBoost-red)

## 📌 Project Overview
This project focuses on predicting customer churn in the e-commerce sector using machine learning. By analyzing customer behavior and transaction history, the model identifies customers who are likely to stop using the service. This insight allows businesses to take proactive measures to retain valuable customers.

**Author:** Muhammad Atif

## 📂 Dataset
The dataset contains information about e-commerce customers, including their demographics, purchase history, and interaction behaviors.

- **Target Variable:** `Churn` (Classifying if a customer left or stayed).
- **Key Features:** Customer demographics, tenure, transaction frequency, and other behavioral metrics.

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:**
  - `pandas` & `numpy` for data manipulation
  - `matplotlib` & `seaborn` for data visualization
  - `scikit-learn` for model building and preprocessing
  - `xgboost` for advanced gradient boosting
  - `pickle` for model serialization

## 📊 Project Workflow

### 1. Data Preprocessing
- **Handling Missing Values:** Utilized `KNNImputer` to fill missing entries efficiently.
- **Outlier Detection & Removal:** Analyzed statistical distributions to remove anomalies that could skew the model.
- **Feature Encoding:** Applied `LabelEncoder` to convert categorical variables into a machine-readable format.

### 2. Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Studied the distribution of individual features.
- **Bivariate & Multivariate Analysis:** Explored relationships between features and the target variable (Churn).

### 3. Model Building
Three different machine learning algorithms were trained and evaluated to find the best performer:
- **Logistic Regression:** For baseline classification.
- **Random Forest Classifier:** For robust, ensemble-based predictions.
- **XGBoost Classifier:** For high-performance gradient boosting.

### 4. Model Evaluation
Models were evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **F1-Score, Recall, and Precision**
- **Cross-Validation** to ensure model stability.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ecommerce-churn-prediction.git](https://github.com/your-username/ecommerce-churn-prediction.git)
