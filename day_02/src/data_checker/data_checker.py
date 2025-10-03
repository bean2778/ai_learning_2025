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

def check_data(data: List[Any]) -> Dict[str, Any]:
    """
    Automatically detect data type and route to appropriate analyzer.
    
    Uses Yamane's formula to determine sample size for type detection.
    
    Args:
        data: List of values to analyze
        
    Returns:
        Dictionary containing analysis results from appropriate analyzer
    """
    if not data or len(data) == 0:
        return {'error': 'empty dataset'}
    
    # Yamane's formula: n = N / (1 + N * e^2)
    # where N = population size, e = margin of error (0.1)
    N = len(data)
    e = 0.1
    sample_size = int(N / (1 + N * (e ** 2)))
    
    # Determine sampling frequency
    if N / sample_size < 2:
        # Check everything
        sample_indices = range(N)
    else:
        # Check every nth element
        frequency = int(N / sample_size)
        sample_indices = range(0, N, frequency)
    
    # Count valid items for each type
    num_count = 0
    temp_count = 0
    cat_count = 0
    
    for i in sample_indices:
        item = data[i]
        
        # Skip None/NaN for type detection
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        
        # Test if temporal (try parsing as datetime)
        if isinstance(item, str):
            # Strings: always test for date parsing
            try:
                pd.to_datetime([item], errors='raise')
                temp_count += 1
            except:
                pass
        elif isinstance(item, int) and item > 946684800:  # Jan 1, 2000 as Unix timestamp
            # Large integers: might be Unix timestamps
            try:
                pd.to_datetime([item], unit='s', errors='raise')
                temp_count += 1
            except:
                pass
        
        # Test if numeric
        if isinstance(item, (int, float, np.number)) and not pd.isna(item):
            num_count += 1
        
        # Test if categorical (string or valid int/float)
        if isinstance(item, str) or isinstance(item, int):
            cat_count += 1
        elif isinstance(item, float) and not pd.isna(item) and item % 1 == 0:
            cat_count += 1
    
    # Determine type (precedence: temporal > numeric > categorical)
    if temp_count >= num_count and temp_count >= cat_count:
        return check_temporal_quality(data)
    elif num_count >= cat_count:
        return check_data_quality(data)
    else:
        return check_categorical_quality(data)


def check_temporal_quality(data: List[str]) -> Dict[str, Any]:
    """
    Perform checks on temporal data

    
    This function is designed with production ML pipelines in mind,
    where data quality is crucial for model performance.

    Args:
        data: List of temporal values to check

    Returns:
        Dictionary containing temporal metrics and flags
    """
    series = pd.Series(data)
    temp_series = pd.to_datetime(series, errors='coerce')
    valid_series = temp_series.dropna()
    if not len(valid_series):
        return {'error' : 'no valid data'}
    
    analysis = {
        'analysis': 'temporal',
        'total values' : len(data),
        'total valid values' : len(valid_series),
        'invalid values' : len(data) - len(valid_series),
        'earliest' : valid_series.min(),
        'latest' : valid_series.max(),
        'frequency/granularity' : valid_series.dt.freq,
        'repeated times' : valid_series.value_counts().to_dict(),
        'diff' : valid_series.diff().to_dict(),
        'gaps_detected' : 0
    }
    time_deltas = valid_series.diff().dropna()

    if time_deltas.nunique() == len(time_deltas):
        analysis['pattern'] = 'irregular'
        analysis['gaps'] = 'N/A - no regular pattern'
    else:
        most_common = time_deltas.mode()[0]
        large_gaps = time_deltas[time_deltas > most_common * 1.5]
        analysis['detected frequency'] = str(most_common)
        if len(large_gaps) > 0:
            # Report AFTER which dates gaps occur
            gap_indices = time_deltas[time_deltas > most_common * 1.5].index
            analysis['gap_after_dates'] = valid_series[gap_indices - 1].tolist()
        analysis['gaps_detected'] = len(large_gaps)


    analysis['repeated times'] = {key: value for key, value in analysis['repeated times'].items() if value > 1}
    return analysis

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
        return { 
                    'analysis' : 'categorical',
                    'error': 'no valid data points' 
                }
    series = pd.Series(valid_data)


    analysis = {
        'analysis' : 'categorical',
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
            'analysis': 'numerical',
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
    dates = [
        '2024-01-01',
        '2024-01-02', 
        '2024-01-03',
        '2024-01-03',
        '2024-01-05',  # Gap!
        '2024-01-06'
    ]
    result = check_temporal_quality(dates)
    print(result)
        # Your test code here
        
if __name__ == "__main__":
    main()