# Final-project Data Science Roles, Skills with Salaries 2025-
![](https://github.com/sripriya1999/Final-project-Data-Science-Job-Postings-with-Salaries-2025-/blob/main/Headerheader.jpg)
> 🔬 A data-driven machine learning project that predicts Data Science Roles, Skills, and Salaries Analysis (2025)

---

## 🧩 Project Overview
It includes job titles, work status (remote, hybrid, on-site), salaries, company size, headquarters location, industry type, and revenue.It also highlights the skills in demand (e.g., Python, SQL, Spark, AWS, machine learning) along with compensation ranges. To analyze job market trends, skill requirements, and salary benchmarks in the data science field, this dataset, which includes global data science job postings for 2025, includes detailed information about job roles, seniority levels, company profiles, industries, and required technical skills.
### 🎯 Objectives
- Perform exploratory data analysis (EDA) on job listings to uncover trends in job titles, skills demand, work status, and compensation.Predict salary ranges based on job title, seniority, skills, and company features (Regression problem).

- Classify job seniority level or work status based on job description and company data (Classification problem).

---

### ▶️ Quick Start
1. Open the notebook:
### ▶️ Run Directly in Google Colab
You can execute the entire workflow without any setup:
🔗 [**Open Project in Colab**](https://colab.research.google.com/drive/197QbMz8pTZObr-Mf-sPzE_RkY_-onCi7?usp=sharing)
#### Codes and Resources Used
- **Editor Used:** Google Colab / Jupyter Notebook  
- **Python Version:** 3.12  
- **Platform:** Google Colab  
- **Environment:** Machine Learning / Salary Comparisoin &Analytics

#### Python Packages Used
- **General Purpose:** `os`, `warnings`, `joblib`, `requests`  
- **Data Manipulation:** `pandas`, `numpy`  
- **Data Visualization:** `matplotlib`, `seaborn`, `plotly`  
- **Machine Learning:** `scikit-learn`, `xgboost`

# Data
The dataset is a crucial part of this project. It Companies and job seekers often face challenges understanding the job market dynamics—such as what skills are in demand, typical salary ranges, or trends in remote work. Using job listing data, this project aims to analyze these trends and predict key job attributes such as salary or seniority level based on job requirements and company details.

I structure this as follows - 

## Source Data
**Author:** Sidraazam

**Source:** Kaggle (on Kaggle) — dataset titled “Data Science Roles, Skills & Salaries 2025”. 

**Timeline:** September 2025

**Shape:** 944 rows × 13 columns. According to one source, the table has 13 columns.
**Target Feature:** Salary

## Data Preprocessing
To make the dataset suitable for modeling:

1.Checked for missing values → Founded

2.Verified duplicate rows → Founded

3.Removed outliers using IQR method

4.Applied skewness correction and transformations (log/square root)

5.Scaled numeric features using StandardScaler / MinMaxScaler

6.Encoded categorical variables using one-hot encoding or label encoding

## 📊📊 Workflow Steps

| Step                       | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| 1️⃣ Load Dataset           | Import raw data into environment                        |
| 2️⃣ Initial EDA            | Analyze distributions, missing values, outliers         |
| 3️⃣ Data Preprocessing     | Handle nulls, outliers, encode categorical features     |
| 4️⃣ Feature Engineering    | Create new features or transform existing ones          |
| 5️⃣ Feature Scaling        | Standardize or normalize numeric features               |
| 6️⃣ Feature Selection      | Select top features using SelectKBest                   |
| 7️⃣ Train/Test Split       | Divide dataset into training and testing sets           |
| 8️⃣ Model Building         | Train multiple machine learning models                  |
| 9️⃣ Hyperparameter Tuning  | Optimize model parameters using RandomizedSearchCV      |
| 🔟 Model Evaluation        | Evaluate using Accuracy, F1, Precision, Recall, AUC-ROC |
| 1️⃣1️⃣ Final Prediction    | Predict diabetes subtype for new patient samples        |
| 1️⃣2️⃣ Future Enhancements | Deep learning, ensemble methods, deployment             |



#### 📊 Dataset
- **Rows × Columns:** Rows × Columns: 944 x 13

- **Features include:** | Job Title | Seniority Level | Work Status | Company Size | Skills | Posting Date | Industry Type |

- **No missing values or duplicates**

#### 🤖 Model Building
#### 🧩 Algorithms Used

The following machine learning algorithms were implemented and compared to identify the best-performing model for multiclass predicts the salary category 

- Random Forest Regressor – tree-based ensemble model.

- XGBoost Regressor – gradient boosting model.

- SVR (Support Vector Regression) – kernel-based regression.

- kNN Regressor – distance-based regression.
#### 🎯 Model Tuning

**Hyperparameter Optimization:** Conducted using GridSearchCV (3-fold cross-validation)


#### 🧠 Model Evaluation Metrics

Each model was evaluated using multiple metrics to ensure balanced performance across all diabetes subtypes:

#### Metric	Description
| Metric                   | Description                                             |
| :----------------------- | :------------------------------------------------------ |
| **Accuracy**             | Overall proportion of correctly classified samples      |
| **Precision**            | Fraction of correctly predicted positive observations   |
| **Recall (Sensitivity)** | Fraction of actual positives correctly identified       |
| **F1-Score**             | Harmonic mean of Precision and Recall                   |
| **AUC-ROC**              | Measures model's ability to distinguish between classes |

#### 🏆 Best Model

#### After comprehensive evaluation:

#### Best Performing Model: 🏅 Random Forest Regression

#### Reason for Selection:

- Highest accuracy and AUC-ROC scores

- Well-balanced precision–recall trade-off

- High interpretability through feature importance

- Robust performance against noise and feature correlations

#### 📈 Sample Prediction
#### Sample Input



| Job Title                 | Seniority Level | Work Status | Company Size | Skills                  | Posting Date | Industry Type |
| ------------------------- | --------------- | ----------- | ------------ | ----------------------- | ------------ | ------------- |
| Data Scientist            | Senior          | Remote      | 500-1000     | Python, SQL, AWS        | 2024-09-15   | Tech          |
| Machine Learning Engineer | Mid             | On-site     | 1000-5000    | Python, TensorFlow, GCP | 2024-08-20   | Finance       |
| Data Analyst              | Entry           | Hybrid      | 50-200       | Excel, SQL, Tableau     | 2024-10-01   | Retail        |

#### Sample Output
| Job Title                 | Predicted Salary  |
| ------------------------- | -------------------- |
| Data Scientist            | High                 |
| Machine Learning Engineer | High                 |
| Data Analyst              | Medium               |



## Final Conclusion

1. A comprehensive machine learning pipeline was developed to predict the salary with including features.

2. Multiple models were trained and evaluated: Random,XGboost,SVM,KNN

3. Random Forest emerged as the best-performing model, achieving 92% accuracy and high performance across Precision, Recall, F1-Score, and AUC-ROC metrics.

4. The dataset confirms that in 2025, data science roles continue to offer strong compensation, driven by seniority, in‑demand skills (e.g., Python, AWS, ML), and industry/organization profile. Industry data shows mid‑level data scientists in India now earn around ₹12 LPA and above. 

5. Globally, experienced data scientists are reaching salaries of USD 150 K or more.
  
6. Therefore, for job‑seekers in data science: accumulating relevant skills, targeting senior roles, and aligning with strong companies/locations remain key strategies for  maximising compensation

7. This project demonstrates the potential of machine learning in healthcare analytics for early detection, personalized risk assessment, and targeted interventions.

8. Future work can include deep learning models, ensemble methods, explainable AI (SHAP/LIME), and deployment as a web/mobile application.

# 🚀 Future Enhancements
Outline potential future work that can be done to extend the project or improve its functionality. This will help others understand the scope of your project and identify areas where they can contribute.

**1.Hyperparameter Optimization:** Use Bayesian optimization or RandomizedSearchCV for more efficient tuning.

**Cross-Validation Improvements:**

- Implement stratified or repeated cross-validation for more robust evaluation.

**2.Feature Engineering:** Add derived features, feature selection 
- Create new features or combine existing ones to better capture relationships.

- Explore interaction terms or polynomial features.

**3.Data Augmentation & Cleaning:**
- Collect more data to improve model generalization.

- Handle outliers or skewed distributions more effectively.

**4.Advanced Models::**
- Test other ensemble models like LightGBM or CatBoost.

- Explore deep learning models if dataset size permits.


# Model Optimization

**Hyperparameter Tuning:**

- Fine-tune parameters such as n_estimators, max_depth, learning_rate, and min_samples_split.

**Regularization**

- Apply techniques like L1/L2 regularization for linear models to reduce overfitting.

**Feature Selection:**

- Remove irrelevant or highly correlated features to improve accuracy and reduce computation.

**Ensemble Techniques:**

- Combine multiple models using stacking or blending for better predictions.

**Scaling and Normalization:**

- Standardize features for distance-based models like kNN or SVR.

# Acknowledgments/References
Acknowledge any contributors, data sources, or other relevant parties who have contributed to the project. This is an excellent way to show your appreciation for those who have helped you along the way.

- Dataset inspired by open-source health data repositories

- README template adapted from Pragyy’s Data Science Readme Template


# License
Specify the license under which your code is released. Moreover, provide the licenses associated with the dataset you are using. This is important for others to know if they want to use or contribute to your project. 

For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
