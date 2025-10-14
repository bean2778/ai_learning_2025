"""
EDA Toolkit - Automated exploratory data analysis functions
"""

from .analyzer import (
    generate_numeric_summary,
    generate_relationship_analysis,
    generate_categorical_analysis
)

__all__ = [
    'generate_numeric_summary',
    'generate_relationship_analysis',
    'generate_categorical_analysis'
]

__version__ = '0.1.0'