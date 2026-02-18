# Medical Insurance Analytics

A comprehensive analysis and predictive modeling framework for medical insurance data.

## 📊 Dataset Overview
The dataset contains 100,000 patient records with 54 features including:
- Demographic information (age, sex, region, education, etc.)
- Health metrics (BMI, blood pressure, HbA1c, LDL, chronic conditions)
- Insurance details (plan type, network tier, deductible, copay)
- Medical history (visits, hospitalizations, procedures)
- Financial information (income, medical costs, premiums)

## 🎯 Project Goals

# Objective
Analyze Medical Insurance Cost Prediction (of 100,000) patients, and classify said patients’ healthcare insurance based on a multitude of factors:
  - Demographic
  - Socioeconomic Status
  - Insurance plans
  - Lifestyle

From a low-high quality insurance standing

# Rationale 
Medical insurance costs are influenced by a wide range of demographic, lifestyle, clinical factors, making cost prediction a challenging and important problem. Accurately predicting medical insurance expenses can help improve risk assessment and support data-driven decision making in healthcare and insurance systems. Due to the complex relationships among these factors, machine learning methods are well suited for this task. 




# Model/Approach
In order to effectively classify our data, applying logistic regression to all relevant variables is a great place to start, also applying regularization on variables such as age, marital status, doctor visits, etc., and including interaction terms for said variables for more flexibility.

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- pip or conda

- ADD ROC AUC AND MCC LATER


### Installation
```bash
# Clone the repository
git clone https://github.com/gyatgitpandaclatt/171PROJECT.git
cd medical-insurance-analytics

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate medical-insurance
