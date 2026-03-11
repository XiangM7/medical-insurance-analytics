"""
Unit tests for data preprocessing.
"""

import pytest # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from src.data_preprocessing import MedicalInsurancePreprocessor

@pytest.fixture
def sample_data():
    data = {
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'sex': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'bmi': [22.5, 24.5, 26.5, 28.5, 30.5],
        'annual_medical_cost': [1000, 2000, 3000, 4000, 5000],
        'annual_premium': [500, 600, 700, 800, 900],
        'hypertension': [0, 1, 0, 1, 0],
        'diabetes': [0, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def test_preprocessor_initialization():
    preprocessor = MedicalInsurancePreprocessor()
    assert preprocessor is not None
    assert len(preprocessor.numerical_features) > 0
    assert len(preprocessor.categorical_features) > 0

def test_fit_transform(sample_data):
    preprocessor = MedicalInsurancePreprocessor()
    processed = preprocessor.fit_transform(sample_data)
    
    assert isinstance(processed, pd.DataFrame)
    assert len(processed) == len(sample_data)
    assert not processed.isnull().any().any()

def test_missing_values_handling():
    data = pd.DataFrame({
        'age': [25, np.nan, 35, 40, 45],
        'income': [50000, 60000, np.nan, 80000, 90000],
        'sex': ['Male', 'Female', 'Male', None, 'Male'],
        'bmi': [22.5, 24.5, 26.5, 28.5, 30.5]
    })
    
    preprocessor = MedicalInsurancePreprocessor()
    processed = preprocessor.fit_transform(data)
    
    assert not processed.isnull().any().any()