import pytest
import pandas as pd
import time
from optimizer.slow_processor import process_transactions_slow
from optimizer.fast_processor import process_transactions_fast

@pytest.fixture
def sample_data():
    """Load sample data"""
    return pd.read_csv('sample_data.csv')

def test_slow_processor_timing(sample_data):
    """Measure how long slow version takes"""
    start = time.time()
    result = process_transactions_slow(sample_data)
    elapsed = time.time() - start
    
    print(f"\nSlow version took: {elapsed:.2f} seconds")
    print(f"Rows processed: {len(result)}")
    print(f"Rows per second: {len(result)/elapsed:.0f}")
    
    assert len(result) == len(sample_data)

def test_outputs_match(sample_data):
    """Ensure fast version produces same results as slow version"""
    slow_result = process_transactions_slow(sample_data)
    fast_result = process_transactions_fast(sample_data)
    
    # Compare all columns
    pd.testing.assert_frame_equal(
        slow_result.sort_index(axis=1),
        fast_result.sort_index(axis=1),
        check_dtype=False  # Important - allows int vs float differences
    )


def test_fast_processor_timing(sample_data):
    """Measure how long slow version takes"""
    start = time.time()
    result = process_transactions_fast(sample_data)
    elapsed = time.time() - start
    
    print(f"\nfast version took: {elapsed:.2f} seconds")
    print(f"Rows processed: {len(result)}")
    print(f"Rows per second: {len(result)/elapsed:.0f}")


def test_performance_comparison(sample_data):
    """Compare slow vs fast versions"""
    import time
    
    # Time slow version
    start = time.time()
    slow_result = process_transactions_slow(sample_data)
    slow_time = time.time() - start
    
    # Time fast version
    start = time.time()
    fast_result = process_transactions_fast(sample_data)
    fast_time = time.time() - start
    
    speedup = slow_time / fast_time
    
    print(f"\nPerformance Comparison:")
    print(f"Slow version: {slow_time:.4f} seconds")
    print(f"Fast version: {fast_time:.4f} seconds")
    print(f"Speedup: {speedup:.1f}x faster")
    
    assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"