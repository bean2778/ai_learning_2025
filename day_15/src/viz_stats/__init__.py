"""Statistical visualization toolkit for ML exploration"""

from .visualizer import (
    plot_age_distribution,
    plot_income_by_segment,
    plot_purchase_distribution,
    plot_income_vs_purchase,
    plot_age_vs_days_since_signup,
    plot_purchase_by_category,
    plot_income_by_segment_violin,
    plot_correlation_matrix
)

__all__ = [
    'plot_age_distribution',
    'plot_income_by_segment',
    'plot_purchase_distribution',
    'plot_income_vs_purchase',
    'plot_age_vs_days_since_signup',
    'plot_purchase_by_category',
    'plot_income_by_segment_violin',
    'plot_correlation_matrix'
]