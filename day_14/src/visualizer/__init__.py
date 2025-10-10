"""
Data Visualization Toolkit for Anomaly Detection

This module provides functions to visualize data quality issues in datasets,
making outliers, missing data, and imbalances immediately obvious.
"""

from .visualizer import (
    plot_amount_distribution,
    plot_time_series,
    plot_age_distribution,
    plot_category_balance,
    create_dashboard
)

__all__ = [
    'plot_amount_distribution',
    'plot_time_series',
    'plot_age_distribution',
    'plot_category_balance',
    'create_dashboard'
]

__version__ = '0.1.0'