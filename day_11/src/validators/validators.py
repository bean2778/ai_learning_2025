from typing import Dict, List, Tuple, Any
import pandas as pd


def validate_schema(df: pd.DataFrame, expected_schema: dict) -> dict:
    """
    Validate that DataFrame matches expected schema.
    
    Args:
        df: DataFrame to validate
        expected_schema: Dict mapping column names to expected dtypes
                        e.g., {'customer_id': 'int64', 'name': 'object'}
    
    Returns:
        Dictionary with keys:
        - 'valid': bool - True if schema matches
        - 'missing_columns': list - Columns in schema but not in df
        - 'extra_columns': list - Columns in df but not in schema
        - 'type_mismatches': list - Tuples of (column, expected, actual)
    """
    schema_match = True
    col_list = df.columns.tolist()
    schema_columns = set(expected_schema.keys())
    df_columns = set(df.columns.tolist())
    missing_columns = schema_columns - df_columns
    extra_columns = df_columns - schema_columns
    mismatch_list = list()
    for col, expected_dtype in expected_schema.items():
        actual_dtype = str(df[col].dtype)
        if col in df.columns:
            if actual_dtype != expected_dtype:
                mismatch_list.append((col, expected_dtype, actual_dtype))
    if missing_columns or extra_columns or mismatch_list:
        schema_match = False
    result_dict = {
        'valid' : schema_match,
        'missing_columns' : missing_columns,
        'extra_columns' : extra_columns,
        'type_mismatches' : mismatch_list
    }
    return result_dict

def validate_business_rules(df: pd.DataFrame) -> dict:
    """
    Validate business rules on customer transaction data.
    
    Checks:
    - No negative amounts
    - Ages between 1-120
    - Email addresses contain '@'
    - Dates not in the future
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with keys:
        - 'valid': bool - True if all rules pass
        - 'violations': dict - Maps rule name to count of violations
        - 'violation_details': dict - Maps rule name to DataFrame of violating rows
    """
    result = {
        'valid': True,
        'violations': {},
        'violation_details': {}
    }
    
    if 'amount' in df.columns:  # Fixed: was 'customer_id'
        result['violation_details']['negative_amounts'] = df[df['amount'] < 0]
        result['violations']['negative_amounts'] = len(result['violation_details']['negative_amounts'])
        if result['violations']['negative_amounts']:
            result['valid'] = False
    
    if 'age' in df.columns:
        result['violation_details']['invalid_ages'] = df[(df['age'] < 1) | (df['age'] > 120)]  # Fixed
        result['violations']['invalid_ages'] = len(result['violation_details']['invalid_ages'])
        if result['violations']['invalid_ages']:
            result['valid'] = False
    
    if 'email' in df.columns:
        result['violation_details']['invalid_emails'] = df[~df['email'].str.contains('@', na=False)]
        result['violations']['invalid_emails'] = len(result['violation_details']['invalid_emails'])
        if result['violations']['invalid_emails']:
            result['valid'] = False
    
    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
        result['violation_details']['future_dates'] = df[df['signup_date'] > pd.Timestamp.now()]
        result['violations']['future_dates'] = len(result['violation_details']['future_dates'])
        if result['violations']['future_dates']:
            result['valid'] = False
    
    return result

def detect_anomalies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Detect statistical outliers in a numeric column using IQR method.
    
    IQR (Interquartile Range) method:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower bound = Q1 - 1.5 * IQR
    - Upper bound = Q3 + 1.5 * IQR
    - Outliers are values outside these bounds
    
    Args:
        df: DataFrame to analyze
        column: Name of numeric column to check for outliers
    
    Returns:
        DataFrame with anomalous rows, with added columns:
        - 'row_index': Original index from input DataFrame
        - 'value': The anomalous value
        - 'reason': Why it's anomalous (e.g., "below lower bound" or "above upper bound")
    """
 # Calculate IQR bounds
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
    
    # Add required columns
    outliers['row_index'] = outliers.index
    outliers['value'] = outliers[column]
    
    # Determine reason
    outliers['reason'] = outliers[column].apply(
        lambda x: 'below lower bound' if x < lower_bound else 'above upper bound'
    )
    
    return outliers
        

"""
    'customer_id': 'int64',
    'name': 'object',
    'age': 'float64',
    'email': 'object',
    'signup_date': 'datetime64[ns]'

"""

df = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})
schema = {'customer_id': 'int64', 'name': 'object'}
set1 = set([1, 2, 3]) 
set2 = set([1, 2, 4])
print(set1 - set2)
print(set2 - set1)
print(df.dtypes)
print(len(df))