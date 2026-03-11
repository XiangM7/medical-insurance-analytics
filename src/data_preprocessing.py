"""
Data preprocessing module for medical insurance dataset.
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore

class MedicalInsurancePreprocessor:    
    def __init__(self):
        self.numerical_features = [
            'age', 'income', 'household_size', 'dependents', 'bmi',
            'medication_count', 'systolic_bp', 'diastolic_bp', 'ldl',
            'hba1c', 'deductible', 'copay', 'policy_term_years',
            'policy_changes_last_2yrs', 'provider_quality', 'risk_score',
            'annual_medical_cost', 'annual_premium', 'monthly_premium',
            'claims_count', 'avg_claim_amount', 'total_claims_paid',
            'chronic_count'
        ]
        
        self.categorical_features = [
            'sex', 'region', 'urban_rural', 'education', 'marital_status',
            'employment_status', 'smoker', 'alcohol_freq', 'plan_type',
            'network_tier'
        ]
        
        self.binary_features = [
            'hypertension', 'diabetes', 'asthma', 'copd',
            'cardiovascular_disease', 'cancer_history', 'kidney_disease',
            'liver_disease', 'arthritis', 'mental_health',
            'is_high_risk', 'had_major_procedure'
        ]
        
        self.procedure_features = [
            'proc_imaging_count', 'proc_surgery_count',
            'proc_physio_count', 'proc_consult_count', 'proc_lab_count'
        ]
        
        self.preprocessor = None
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features),
            ('bin', 'passthrough', self.binary_features),
            ('proc', 'passthrough', self.procedure_features)
        ])
        
        return pd.DataFrame(
            self.preprocessor.fit_transform(df),
            columns=self._get_feature_names()
        )
    
    def _get_feature_names(self) -> list:
        feature_names = []
        
        # Numerical features
        feature_names.extend(self.numerical_features)
        
        # Categorical features (will be expanded by OneHotEncoder)
        feature_names.extend([f'cat_{i}' for i in range(len(self.categorical_features) * 5)])  # Approximate
        
        # Binary and procedure features
        feature_names.extend(self.binary_features)
        feature_names.extend(self.procedure_features)
        
        return feature_names