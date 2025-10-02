import sys
try:
    import logging
except ImportError:
    print("missing logging lib")
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    import numpy as np
    import pandas as pd
    import math
    from typing import List, Dict, Any
except ImportError:
    logging.error("Missing numpy, pandas, or typing")
    sys.exit(1)


def check_categorical_quality(data: List[str]) -> Dict[str, Any]:
    """
    Perform checks on categorical data

    
    This function is designed with production ML pipelines in mind,
    where data quality is crucial for model performance.

    Args:
        data: List of categorical values to check

    Returns:
        Dictionary containing categorical metrics and flags
    """

    valid_data = []
    empty_string_count = 0
    invalid_type = 0
    str_count = 0
    num_count = 0
    none_count = 0
    floats = False

    for item in data:
        if isinstance(item, float):
            if item % 1 == 0:
                valid_data.append(item)
                num_count += 1
            else:
                invalid_type += 1
                floats = True
        elif isinstance(item, int):
            valid_data.append(item)
            num_count += 1
        elif item is None or pd.isna(item):
            none_count +=1
        elif item == "":
            empty_string_count += 1
        elif isinstance(item, str):
            valid_data.append(item)
            str_count += 1
        else:
            invalid_type +=1
    # cardinality,  most/least common, missing val count, distribution

    if not valid_data:
        return { 'error': 'no valid data points' }
    series = pd.Series(valid_data)


    analysis = {
        'total_values': len(data),
        'valid_categories': len(valid_data),
        'missing_none': none_count,
        'missing_empty_string': empty_string_count,
        'invalid_type_count': invalid_type,
        'frequency distribution' : series.value_counts().to_dict(),
        'unique count' : series.nunique(),
        'most common': series.value_counts().index[0],
        'least common': series.value_counts().index[-1]
    }

    warnings = []
    if floats:
        warnings.append('Float detected, did you want numerical analysis?')
    elif valid_data and (str_count <= len(valid_data) / 2):
        warnings.append('not many strings, did you want numerical analysis?')
    if len(warnings):
        analysis['warnings'] = warnings

    return analysis
        

        
        


def check_data_quality(data: List[float]) -> Dict[str, Any]:
    """
    Perform basic data quality checks on numerical data.
    
    This function is designed with production ML pipelines in mind,
    where data quality is crucial for model performance.
    
    Args:
        data: List of numerical values to check
        
    Returns:
        Dictionary containing quality metrics and flags
    """
    # Categorize all data in one pass for transparency
    valid_data = []
    excluded_non_numeric = []
    excluded_nan_none = []
    
    for item in data:
        if item is None or pd.isna(item):
            excluded_nan_none.append(item)
        elif not isinstance(item, (int, float, np.number)):
            excluded_non_numeric.append(item)
        else:
            valid_data.append(item)
    
    if not valid_data:
        return {
            'data points': len(data),
            'valid data points': 0,
            'error': 'No valid data points to analyze',
            'excluded_non_numeric': excluded_non_numeric,
            'excluded_nan_none': excluded_nan_none
        }
    
    try:
        logging.info(f"Processing {len(valid_data)} valid data points")
        analysis = {
            'data points': len(data),
            'valid data points': len(valid_data),
            'std': np.std(valid_data),
            'mean': np.mean(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'outliers': find_outlier(valid_data),
            'excluded_non_numeric': excluded_non_numeric,
            'excluded_nan_none': excluded_nan_none
        }
    except Exception as e:
        logging.error(f"Error calculating statistics: {e}")
        return {
            'data points': len(data),
            'valid data points': len(valid_data),
            'error': f'Calculation error: {str(e)}',
            'excluded_non_numeric': excluded_non_numeric,
            'excluded_nan_none': excluded_nan_none
        }
    
    return analysis

def find_outlier(cdata: List[float]) -> List[float]:
    """
    FIne outliers by getting std dev and then doing some arithmatic

    Args:
        cdata: List of vals that has been cleaned of Nones

    Returns:
        List of vals of outliers
    
    """
    std = np.std(cdata)
    mean = np.mean(cdata)
    result = [x for x in cdata if (x < (mean - 2 * std)) or (x > (mean + 2 * std))]
    return result

def analyze_csv_file(filepath):
    """
    Integration point: Read file, analyze, return report
    This is the kind of workflow you'll have in real ML pipelines
    """
    
    try:
        # Step 1: Read file
        df = pd.read_csv(filepath)
        
        # Step 2: Extract numeric column (assume first column)
        data = pd.to_numeric(df.iloc[:, 0], errors='coerce').tolist()
        
        # Step 3: Analyze
        result = check_data_quality(data)
        
        # Step 4: Add metadata
        result['source_file'] = filepath
        result['total_rows'] = len(df)
        
        return result
        
    except FileNotFoundError:
        return {'error': f'File not found: {filepath}'}
    except Exception as e:
        return {'error': f'Failed to process file: {str(e)}'}   
    


def main():
    """Main function to test data quality checker."""
    # # Test with sample data
    # test_data = [1, 2, 3, 4, 5, 100, 2, 3, 4, None, 5]
    
    # print("🔍 Data Quality Report")
    # print("=" * 30)
    # print(check_data_quality(test_data))

    # edge_cases = [
    #     [],  # Empty list
    #     [None, None, None],  # All None
    #     [5, 5, 5, 5, 5],  # No variation
    #     [1, 2, float('nan'), 4, 5]  # With NaN
    # ]

    # for i, case in enumerate(edge_cases):
    #     print(f"\n🧪 Edge Case {i+1}: {case}")
    #     result = check_data_quality(case)
    #     print(result)

    data = ['a', 'b', 'b', 'b', 'c', 'c']
    result = check_categorical_quality(data)
    print(result)
        # Your test code here
        
if __name__ == "__main__":
    main()