# Medical Insurance Analytics and Predictive Modeling

A machine learning project for analyzing medical insurance data, predicting healthcare-related costs, and classifying insurance quality based on demographic, clinical, lifestyle, socioeconomic, and insurance-plan features.

## Overview

Medical insurance cost is affected by many interacting factors, including patient demographics, health conditions, lifestyle indicators, insurance plan structure, medical history, and financial background. This project explores how machine learning can be used to model those relationships and support data-driven risk assessment in healthcare and insurance systems.

The project uses a dataset of 100,000 patient records with 54 features, including demographic information, health metrics, insurance details, medical history, and financial variables.

## Project Goals

The main goals of this project are to:

* Analyze the key factors that influence medical insurance cost and coverage quality
* Build a machine learning pipeline for structured healthcare and insurance data
* Frame the task as both cost prediction and insurance quality classification
* Apply logistic regression and regularization-based modeling as interpretable baseline methods
* Explore how demographic, socioeconomic, lifestyle, clinical, and plan-level features affect predicted outcomes

## Dataset

The dataset contains patient-level records with features such as:

* Demographics: age, sex, region, education, marital status
* Health metrics: BMI, blood pressure, HbA1c, LDL, chronic conditions
* Insurance details: plan type, network tier, deductible, copay
* Medical history: doctor visits, hospitalizations, procedures
* Financial information: income, medical costs, premiums

The data is used to study how patient characteristics and insurance-plan structures relate to predicted medical expenses and insurance quality.

## Machine Learning Approach

This project focuses on interpretable machine learning methods for structured tabular data.

The modeling approach includes:

* Data preprocessing and feature cleaning
* Exploratory data analysis of patient, health, and insurance variables
* Feature selection for relevant demographic, clinical, lifestyle, and financial factors
* Logistic regression as a baseline classification model
* Regularization to reduce overfitting and improve generalization
* Interaction terms to capture relationships between important variables
* Evaluation of model behavior and feature influence

Logistic regression was selected as a starting model because it is interpretable, efficient, and suitable for understanding how individual variables contribute to prediction outcomes.

## Technical Skills Demonstrated

This project demonstrates experience with:

* Python-based machine learning
* Structured data analysis
* Feature engineering
* Logistic regression
* Regularization
* Classification problem framing
* Healthcare and insurance data interpretation
* Model evaluation planning
* Data-driven decision support

## Why This Project Matters

Healthcare and insurance decisions often depend on complex combinations of patient risk, medical history, socioeconomic context, and plan design. Machine learning can help identify important patterns in these variables and support more consistent risk assessment.

This project is designed to show how machine learning can be applied to a realistic, high-dimensional tabular dataset while keeping the model interpretable and explainable.

## My Contributions

My main contributions included:

* Framing the machine learning problem around medical cost prediction and insurance quality classification
* Analyzing relevant demographic, socioeconomic, lifestyle, clinical, and insurance-plan variables
* Designing the baseline modeling approach using logistic regression, regularization, and interaction terms
* Helping structure the project documentation, modeling rationale, and setup process
* Connecting the technical model design to a practical healthcare risk-assessment use case

## Setup

Clone the repository:

```bash
git clone https://github.com/XiangM7/medical-insurance-analytics.git
cd medical-insurance-analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If using conda:

```bash
conda env create -f environment.yml
conda activate medical-insurance-analytics
```

## Usage

Run the analysis notebooks or Python scripts included in the repository to explore the dataset, preprocess features, train baseline models, and evaluate classification or prediction performance.

Typical workflow:

```text
1. Load and inspect the dataset
2. Clean and preprocess features
3. Perform exploratory data analysis
4. Select relevant features
5. Train baseline logistic regression models
6. Evaluate model behavior
7. Interpret feature effects and limitations
```

## Future Improvements

Future versions of this project could include:

* Comparing logistic regression with tree-based models such as random forest or gradient boosting
* Adding cross-validation and hyperparameter tuning
* Improving feature importance analysis
* Adding visualizations for model interpretation
* Testing additional classification and regression metrics
* Building a cleaner end-to-end ML pipeline

## Author

Xiang Mao
B.S. Computer Engineering, University of California, Davis
GitHub: https://github.com/XiangM7
LinkedIn: https://www.linkedin.com/in/xiang-mao-78ab73301/
