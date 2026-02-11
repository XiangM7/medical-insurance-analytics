
"""
Main pipeline script for medical insurance analytics.
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import yaml # type: ignore
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import MedicalInsurancePreprocessor
from visualization import MedicalInsuranceVisualizer

def load_config(config_path: str = 'config/config.yaml'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main pipeline function."""
    # Load configuration
    config = load_config()
    
    # Set paths
    data_dir = Path(config['paths']['data_dir'])
    raw_data_path = data_dir / 'raw' / config['files']['raw_data']
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    
    # Data overview
    print("\nData Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df[['age', 'income', 'bmi', 'annual_medical_cost', 'annual_premium']].describe())
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = MedicalInsurancePreprocessor()
    processed_df = preprocessor.fit_transform(df)
    print(f"Processed shape: {processed_df.shape}")
    
    # Save processed data
    processed_path = data_dir / 'processed' / 'medical_insurance_processed.csv'
    processed_df.to_csv(processed_path, index=False)
    print(f"Saved processed data to: {processed_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = MedicalInsuranceVisualizer()
    
    # Static plots
    fig_static = visualizer.plot_cost_distribution(df)
    fig_static.savefig('reports/figures/cost_distribution.png', dpi=300, bbox_inches='tight')
    
    # Interactive dashboard
    fig_interactive = visualizer.plot_interactive_dashboard(df)
    fig_interactive.write_html('reports/figures/interactive_dashboard.html')
    
    print("\nAnalysis complete!")
    print(f"Reports saved to: reports/figures/")

if __name__ == "__main__":
    main()