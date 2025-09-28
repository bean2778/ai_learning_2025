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
    # Your code here
    clean_data = [x for x in data if not pd.isna(x)]
    nonNums = [x for x in data if not isinstance(x, (int, float,np.number))]
    clean_data = [x for x in clean_data if isinstance(x, (int, float, np.number))]
    clean_data = [x for x in clean_data if not math.isnan(x)]


    if not clean_data:  # Handle empty data
        return {
            'data points': len(data),
            'valid data points': 0,
            'error': 'No valid data points to analyze',
            'excluded data' : nonNums
        }
    
    try:
        print(clean_data)
        analysis = {
            'data points': len(data),
            'valid data points' : len(clean_data),
            'std' : np.std(clean_data),
            'mean' : np.mean(clean_data),
            'min' : np.min(clean_data),
            'max' : np.max(clean_data), 
            'outliers' : find_outlier(clean_data),
            'excluded data' : nonNums
        }
    except TypeError:
        logging.error("incompatible data in dataset")
        return {
            'data points': len(data),
            'valid data points': len(clean_data),
            'error': 'Type error thrown. Check types in data set',
            'excluded data' : nonNums
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

def main():
    """Main function to test data quality checker."""
    # Test with sample data
    test_data = [1, 2, 3, 4, 5, 100, 2, 3, 4, None, 5]
    
    print("🔍 Data Quality Report")
    print("=" * 30)
    print(check_data_quality(test_data))

    edge_cases = [
        [],  # Empty list
        [None, None, None],  # All None
        [5, 5, 5, 5, 5],  # No variation
        [1, 2, float('nan'), 4, 5]  # With NaN
    ]

    for i, case in enumerate(edge_cases):
        print(f"\n🧪 Edge Case {i+1}: {case}")
        result = check_data_quality(case)
        print(result)
        
        # Your test code here
        
if __name__ == "__main__":
    main()