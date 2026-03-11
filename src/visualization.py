"""
Visualization module for medical insurance data.
"""

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd # type: ignore

class MedicalInsuranceVisualizer:
    
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
        
    def plot_cost_distribution(self, df: pd.DataFrame):
        """Plot distribution of medical costs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Annual medical cost
        axes[0, 0].hist(df['annual_medical_cost'], bins=50, color=self.colors[0])
        axes[0, 0].set_title('Distribution of Annual Medical Costs')
        axes[0, 0].set_xlabel('Annual Medical Cost ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Annual premium
        axes[0, 1].hist(df['annual_premium'], bins=50, color=self.colors[1])
        axes[0, 1].set_title('Distribution of Annual Premiums')
        axes[0, 1].set_xlabel('Annual Premium ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Cost vs Premium scatter
        axes[1, 0].scatter(df['annual_medical_cost'], df['annual_premium'], 
                          alpha=0.5, color=self.colors[2])
        axes[1, 0].set_title('Medical Cost vs Premium')
        axes[1, 0].set_xlabel('Annual Medical Cost ($)')
        axes[1, 0].set_ylabel('Annual Premium ($)')
        
        # Cost by age
        age_groups = pd.cut(df['age'], bins=5)
        cost_by_age = df.groupby(age_groups)['annual_medical_cost'].mean()
        axes[1, 1].bar(range(len(cost_by_age)), cost_by_age, color=self.colors[3])
        axes[1, 1].set_title('Average Medical Cost by Age Group')
        axes[1, 1].set_xlabel('Age Group')
        axes[1, 1].set_ylabel('Average Medical Cost ($)')
        axes[1, 1].set_xticks(range(len(cost_by_age)))
        axes[1, 1].set_xticklabels([str(x) for x in cost_by_age.index], rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_dashboard(self, df: pd.DataFrame):
        """Create interactive Plotly dashboard."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Medical Cost by Age', 'Premium by Plan Type',
                          'Chronic Conditions Distribution', 'Cost by Region',
                          'Risk Score vs Cost', 'BMI Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'histogram'}]]
        )
        
        # Medical Cost by Age
        fig.add_trace(
            go.Scatter(x=df['age'], y=df['annual_medical_cost'],
                      mode='markers', name='Medical Cost',
                      marker=dict(color=df['bmi'], colorscale='Viridis',
                                showscale=True, colorbar=dict(title='BMI'))),
            row=1, col=1
        )
        
        # Premium by Plan Type
        for plan_type in df['plan_type'].unique():
            plan_data = df[df['plan_type'] == plan_type]
            fig.add_trace(
                go.Box(y=plan_data['annual_premium'], name=plan_type,
                      boxpoints='outliers'),
                row=1, col=2
            )
        
        # Chronic Conditions Distribution
        chronic_cols = ['hypertension', 'diabetes', 'asthma', 'copd',
                       'cardiovascular_disease', 'cancer_history']
        chronic_counts = df[chronic_cols].sum().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=chronic_counts.values, y=chronic_counts.index,
                  orientation='h', name='Chronic Conditions'),
            row=1, col=3
        )
        
        # Cost by Region
        region_costs = df.groupby('region')['annual_medical_cost'].mean().sort_values()
        fig.add_trace(
            go.Bar(x=region_costs.index, y=region_costs.values,
                  name='Region Costs'),
            row=2, col=1
        )
        
        # Risk Score vs Cost
        fig.add_trace(
            go.Scatter(x=df['risk_score'], y=df['annual_medical_cost'],
                      mode='markers', name='Risk vs Cost',
                      marker=dict(size=8, opacity=0.6)),
            row=2, col=2
        )
        
        # BMI Distribution
        fig.add_trace(
            go.Histogram(x=df['bmi'], nbinsx=30, name='BMI Distribution'),
            row=2, col=3
        )
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Medical Insurance Analytics Dashboard")
        return fig
    #!/usr/bin/env python3