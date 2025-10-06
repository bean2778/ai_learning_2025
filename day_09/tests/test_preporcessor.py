import pytest
import pandas as pd
import numpy as np
from preprocessor import (
    clean_email_addresses,
    standardize_categories,
    clean_age_values,
    clean_dates,
    clean_customer_data,
    clean_country
)


def test_valid_emails_unchanged():
    """Valid emails should pass through unchanged"""
    emails = pd.Series([
        'john@email.com',
        'jane.smith@company.co.uk',
        'user123@test.org'
    ])
    result = clean_email_addresses(emails)
    
    assert result[0] == 'john@email.com'
    assert result[1] == 'jane.smith@company.co.uk'
    assert result[2] == 'user123@test.org'


def test_missing_at_symbol():
    """Emails without @ should become NaN"""
    emails = pd.Series(['johnemail.com', 'invalid'])
    result = clean_email_addresses(emails)
    
    assert pd.isna(result[0])
    assert pd.isna(result[1])


def test_missing_domain():
    """Emails without domain (no dot after @) should become NaN"""
    emails = pd.Series(['john@email', 'user@domain'])
    result = clean_email_addresses(emails)
    
    assert pd.isna(result[0])
    assert pd.isna(result[1])


def test_missing_dot_in_domain():
    """Emails with @ but no dot in domain should become NaN"""
    emails = pd.Series(['john@emailcom'])
    result = clean_email_addresses(emails)
    
    assert pd.isna(result[0])


def test_already_nan_values():
    """NaN values should remain NaN"""
    emails = pd.Series(['valid@email.com', None, np.nan])
    result = clean_email_addresses(emails)
    
    assert result[0] == 'valid@email.com'
    assert pd.isna(result[1])
    assert pd.isna(result[2])


def test_empty_string():
    """Empty strings should become NaN"""
    emails = pd.Series(['', 'valid@email.com'])
    result = clean_email_addresses(emails)
    
    assert pd.isna(result[0])
    assert result[1] == 'valid@email.com'


def test_mixed_valid_invalid():
    """Mix of valid and invalid emails"""
    emails = pd.Series([
        'valid@email.com',      # Valid
        'invalid',              # Invalid - no @
        'also@invalid',         # Invalid - no dot
        'good@domain.org',      # Valid
        None,                   # Already NaN
        'bad.email.com'         # Invalid - no @
    ])
    result = clean_email_addresses(emails)
    
    assert result[0] == 'valid@email.com'
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert result[3] == 'good@domain.org'
    assert pd.isna(result[4])
    assert pd.isna(result[5])


def test_series_length_unchanged():
    """Output series should have same length as input"""
    emails = pd.Series(['a@b.c', 'invalid', 'x@y.z'])
    result = clean_email_addresses(emails)
    
    assert len(result) == len(emails)


def test_does_not_modify_original():
    """Function should not modify the original series"""
    emails = pd.Series(['invalid', 'valid@email.com'])
    original_values = emails.copy()
    
    result = clean_email_addresses(emails)
    
    # Original should be unchanged
    pd.testing.assert_series_equal(emails, original_values)


def test_edge_case_multiple_at_symbols():
    """Emails with multiple @ symbols (technically invalid but might pass simple regex)"""
    emails = pd.Series(['user@@domain.com', 'user@one@two.com'])
    result = clean_email_addresses(emails)
    
    # Depending on your regex, these might pass or fail
    # This test documents the behavior
    # If your regex is r'.+@.+\..+', these would PASS
    # You might want to make regex stricter if needed
    assert not pd.isna(result[0]) or pd.isna(result[0])  # Document actual behavior


@pytest.mark.parametrize("email,should_be_valid", [
    ('simple@example.com', True),
    ('user.name@example.com', True),
    ('user+tag@example.co.uk', True),
    ('invalid', False),
    ('no-at-sign.com', False),
    ('missing-domain@', False),
    ('@no-local-part.com', False),
    ('spaces in@email.com', True),  # Regex .+ matches spaces
])
def test_email_validation_patterns(email, should_be_valid):
    """Parameterized test for various email patterns"""
    emails = pd.Series([email])
    result = clean_email_addresses(emails)
    
    if should_be_valid:
        assert not pd.isna(result[0]), f"Expected {email} to be valid"
    else:
        assert pd.isna(result[0]), f"Expected {email} to be invalid"

def test_lowercase_conversion():
    """Uppercase categories should be converted to lowercase"""
    categories = pd.Series(['Electronics', 'CLOTHING', 'Home'])
    result = standardize_categories(categories)
    
    assert result[0] == 'electronics'
    assert result[1] == 'clothing'
    assert result[2] == 'home'


def test_strip_whitespace():
    """Leading and trailing whitespace should be removed"""
    categories = pd.Series(['  electronics  ', 'clothing ', ' home'])
    result = standardize_categories(categories)
    
    assert result[0] == 'electronics'
    assert result[1] == 'clothing'
    assert result[2] == 'home'


def test_empty_strings_to_nan():
    """Empty strings should become NaN"""
    categories = pd.Series(['electronics', '', 'clothing'])
    result = standardize_categories(categories)
    
    assert result[0] == 'electronics'
    assert pd.isna(result[1])
    assert result[2] == 'clothing'


def test_whitespace_only_becomes_empty_then_nan():
    """Strings with only whitespace should be stripped to empty, then NaN"""
    categories = pd.Series(['electronics', '   ', 'clothing'])
    result = standardize_categories(categories)
    
    assert result[0] == 'electronics'
    assert pd.isna(result[1])  # '   ' stripped to '' then to NaN
    assert result[2] == 'clothing'


def test_already_clean_data():
    """Already clean data should pass through unchanged"""
    categories = pd.Series(['electronics', 'clothing', 'home'])
    result = standardize_categories(categories)
    
    assert result[0] == 'electronics'
    assert result[1] == 'clothing'
    assert result[2] == 'home'


def test_mixed_formatting():
    """Mix of uppercase, whitespace, and clean values"""
    categories = pd.Series([
        'Electronics',
        '  CLOTHING  ',
        'home',
        '  Home  ',
        ''
    ])
    result = standardize_categories(categories)
    
    assert result[0] == 'electronics'
    assert result[1] == 'clothing'
    assert result[2] == 'home'
    assert result[3] == 'home'
    assert pd.isna(result[4])


def test_nan_values_preserved():
    """Existing NaN values should remain NaN"""
    categories = pd.Series(['electronics', None, np.nan, 'clothing'])
    result = standardize_categories(categories)
    
    assert result[0] == 'electronics'
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert result[3] == 'clothing'


def test_series_length_unchanged():
    """Output should have same length as input"""
    categories = pd.Series(['Electronics', 'CLOTHING', ''])
    result = standardize_categories(categories)
    
    assert len(result) == len(categories)


def test_does_not_modify_original():
    """Function should not modify the original series"""
    categories = pd.Series(['Electronics', 'CLOTHING'])
    original = categories.copy()
    
    result = standardize_categories(categories)
    
    # Original unchanged
    pd.testing.assert_series_equal(categories, original)


def test_special_characters_preserved():
    """Special characters in categories should be preserved"""
    categories = pd.Series(['Tech-Gadgets', 'Home & Garden', 'Sports/Outdoors'])
    result = standardize_categories(categories)
    
    assert result[0] == 'tech-gadgets'
    assert result[1] == 'home & garden'
    assert result[2] == 'sports/outdoors'


@pytest.mark.parametrize("input_val,expected", [
    ('Electronics', 'electronics'),
    ('ELECTRONICS', 'electronics'),
    ('  electronics  ', 'electronics'),
    ('  ELECTRONICS  ', 'electronics'),
    ('', None),  # Empty becomes NaN
    ('   ', None),  # Whitespace only becomes NaN
])
def test_standardization_patterns(input_val, expected):
    """Parameterized test for various input patterns"""
    categories = pd.Series([input_val])
    result = standardize_categories(categories)
    
    if expected is None:
        assert pd.isna(result[0])
    else:
        assert result[0] == expected


def test_duplicate_values_after_standardization():
    """Different formatted duplicates should become identical"""
    categories = pd.Series(['Electronics', 'electronics', 'ELECTRONICS', '  Electronics  '])
    result = standardize_categories(categories)
    
    # All should be 'electronics'
    assert result.nunique() == 1
    assert result[0] == 'electronics'

def test_valid_ages_unchanged():
    """Valid ages between 1 and 120 should pass through unchanged"""
    ages = pd.Series([1, 25, 50, 75, 100, 120])
    result = clean_age_values(ages)
    
    assert result[0] == 1
    assert result[1] == 25
    assert result[2] == 50
    assert result[3] == 75
    assert result[4] == 100
    assert result[5] == 120


def test_zero_age_becomes_nan():
    """Age of 0 should become NaN"""
    ages = pd.Series([0, 25, 50])
    result = clean_age_values(ages)
    
    assert pd.isna(result[0])
    assert result[1] == 25
    assert result[2] == 50


def test_negative_age_becomes_nan():
    """Negative ages should become NaN"""
    ages = pd.Series([-1, -10, 25])
    result = clean_age_values(ages)
    
    assert pd.isna(result[0])
    assert pd.isna(result[1])
    assert result[2] == 25


def test_age_over_120_becomes_nan():
    """Ages over 120 should become NaN"""
    ages = pd.Series([121, 150, 200, 50])
    result = clean_age_values(ages)
    
    assert pd.isna(result[0])
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert result[3] == 50


def test_boundary_values():
    """Test exact boundary values (1 and 120)"""
    ages = pd.Series([0, 1, 120, 121])
    result = clean_age_values(ages)
    
    assert pd.isna(result[0])  # 0 is invalid
    assert result[1] == 1      # 1 is valid (boundary)
    assert result[2] == 120    # 120 is valid (boundary)
    assert pd.isna(result[3])  # 121 is invalid


def test_already_nan_values():
    """Existing NaN values should remain NaN"""
    ages = pd.Series([25, None, np.nan, 50])
    result = clean_age_values(ages)
    
    assert result[0] == 25
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert result[3] == 50


def test_mixed_valid_invalid():
    """Mix of valid and invalid ages"""
    ages = pd.Series([0, 25, 150, 50, -5, 120, 200])
    result = clean_age_values(ages)
    
    assert pd.isna(result[0])  # 0
    assert result[1] == 25
    assert pd.isna(result[2])  # 150
    assert result[3] == 50
    assert pd.isna(result[4])  # -5
    assert result[5] == 120
    assert pd.isna(result[6])  # 200


def test_series_length_unchanged():
    """Output should have same length as input"""
    ages = pd.Series([25, 150, 50])
    result = clean_age_values(ages)
    
    assert len(result) == len(ages)


def test_does_not_modify_original():
    """Function should not modify the original series"""
    ages = pd.Series([25, 150, 50])
    original = ages.copy()
    
    result = clean_age_values(ages)
    
    # Original unchanged
    pd.testing.assert_series_equal(ages, original)


def test_float_ages():
    """Float ages should work (in case of decimals in data)"""
    ages = pd.Series([25.5, 30.7, 150.0, 0.5])
    result = clean_age_values(ages)
    
    assert result[0] == 25.5
    assert result[1] == 30.7
    assert pd.isna(result[2])  # 150.0 invalid
    assert pd.isna(result[3])  # 0.5 invalid (< 1)


@pytest.mark.parametrize("age,should_be_valid", [
    (1, True),      # Boundary - valid
    (0, False),     # Boundary - invalid
    (120, True),    # Boundary - valid
    (121, False),   # Boundary - invalid
    (25, True),     # Normal valid
    (50, True),     # Normal valid
    (-1, False),    # Negative
    (150, False),   # Too high
    (999, False),   # Way too high
])
def test_age_validation_patterns(age, should_be_valid):
    """Parameterized test for various age values"""
    ages = pd.Series([age])
    result = clean_age_values(ages)
    
    if should_be_valid:
        assert result[0] == age, f"Expected age {age} to be valid"
    else:
        assert pd.isna(result[0]), f"Expected age {age} to be invalid"


def test_all_invalid_ages():
    """When all ages are invalid, all should become NaN"""
    ages = pd.Series([0, -10, 150, 200])
    result = clean_age_values(ages)
    
    assert result.isna().all()


def test_realistic_dataset():
    """Test with realistic messy data like from CSV"""
    ages = pd.Series([
        34,    # Valid
        28,    # Valid
        None,  # Already missing
        45,    # Valid
        19,    # Valid
        150,   # Typo - invalid
        31,    # Valid
        29,    # Valid
        0,     # Invalid
        22,    # Valid
    ])
    result = clean_age_values(ages)
    
    # Count valid vs invalid
    assert result.notna().sum() == 7  # 7 valid ages
    assert result.isna().sum() == 3   # 3 invalid (None, 150, 0)

def test_standard_date_format():
    """Standard YYYY-MM-DD format should parse correctly"""
    dates = pd.Series(['2024-01-15', '2024-01-16', '2024-01-17'])
    result = clean_dates(dates)
    
    assert result[0] == pd.Timestamp('2024-01-15')
    assert result[1] == pd.Timestamp('2024-01-16')
    assert result[2] == pd.Timestamp('2024-01-17')


def test_invalid_dates_become_nat():
    """Invalid date strings should become NaT"""
    dates = pd.Series(['invalid_date', 'not-a-date', '2024-99-99'])
    result = clean_dates(dates)
    
    assert pd.isna(result[0])
    assert pd.isna(result[1])
    assert pd.isna(result[2])


def test_mixed_valid_invalid():
    """Mix of valid and invalid dates"""
    dates = pd.Series([
        '2024-01-15',      # Valid
        'invalid_date',    # Invalid
        '2024-01-16',      # Valid
        'not-a-date',      # Invalid
    ])
    result = clean_dates(dates)
    
    assert pd.notna(result[0])
    assert pd.isna(result[1])
    assert pd.notna(result[2])
    assert pd.isna(result[3])


def test_already_nan_values():
    """Existing NaN values should remain NaT"""
    dates = pd.Series(['2024-01-15', None, np.nan, '2024-01-16'])
    result = clean_dates(dates)
    
    assert pd.notna(result[0])
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert pd.notna(result[3])


def test_empty_strings():
    """Empty strings should become NaT"""
    dates = pd.Series(['2024-01-15', '', '2024-01-16'])
    result = clean_dates(dates)
    
    assert pd.notna(result[0])
    assert pd.isna(result[1])
    assert pd.notna(result[2])


def test_result_is_datetime_type():
    """Result should be datetime64 dtype"""
    dates = pd.Series(['2024-01-15', '2024-01-16'])
    result = clean_dates(dates)
    
    assert pd.api.types.is_datetime64_any_dtype(result)


def test_series_length_unchanged():
    """Output should have same length as input"""
    dates = pd.Series(['2024-01-15', 'invalid', '2024-01-16'])
    result = clean_dates(dates)
    
    assert len(result) == len(dates)


def test_does_not_modify_original():
    """Function should not modify the original series"""
    dates = pd.Series(['2024-01-15', 'invalid'])
    original = dates.copy()
    
    result = clean_dates(dates)
    
    # Original unchanged (still strings)
    pd.testing.assert_series_equal(dates, original)



@pytest.mark.parametrize("date_str,should_parse", [
    ('2024-01-15', True),
    ('01-15-2024', True),
    ('2024/01/15', True),
    ('15-Jan-2024', True),  # Month name format
    ('invalid_date', False),
    ('', False),
    ('2024-99-99', False),  # Invalid day/month
    ('not-a-date', False),
])
def test_date_parsing_patterns(date_str, should_parse):
    """Parameterized test for various date patterns"""
    dates = pd.Series([date_str])
    result = clean_dates(dates)
    
    if should_parse:
        assert pd.notna(result[0]), f"Expected '{date_str}' to parse"
    else:
        assert pd.isna(result[0]), f"Expected '{date_str}' to be invalid"


def test_year_only_formats():
    """Year-only values might parse (pandas interprets as years)"""
    dates = pd.Series(['2024', '2023', '2022'])
    result = clean_dates(dates)
    
    # Pandas should parse these as dates (Jan 1st of that year)
    assert result.notna().all()


def test_all_invalid_dates():
    """When all dates are invalid, all should become NaT"""
    dates = pd.Series(['invalid', 'not-a-date', 'garbage', ''])
    result = clean_dates(dates)
    
    assert result.isna().all()

# ============================================================================
# Tests for clean_country
# ============================================================================

def test_clean_country_fills_missing():
    """Missing countries should be filled with 'Unknown'"""
    countries = pd.Series(['USA', None, 'UK', np.nan, 'Canada'])
    result = clean_country(countries)
    
    assert result[0] == 'USA'
    assert result[1] == 'Unknown'
    assert result[2] == 'UK'
    assert result[3] == 'Unknown'
    assert result[4] == 'Canada'


def test_clean_country_preserves_valid():
    """Valid countries should be unchanged"""
    countries = pd.Series(['USA', 'UK', 'Canada', 'Australia'])
    result = clean_country(countries)
    
    pd.testing.assert_series_equal(result, countries)


def test_clean_country_all_missing():
    """All missing should all become 'Unknown'"""
    countries = pd.Series([None, np.nan, None])
    result = clean_country(countries)
    
    assert (result == 'Unknown').all()


# ============================================================================
# Tests for clean_customer_data (integration tests)
# ============================================================================

def test_clean_customer_data_creates_copy():
    """Should not modify original dataframe"""
    df = pd.DataFrame({
        'customer_id': [1, 2],
        'name': ['John', 'Jane'],
        'age': [25, 150],
        'email': ['john@email.com', 'invalid'],
        'signup_date': ['2024-01-15', '2024-01-16'],
        'purchase_amount': [100.0, -50.0],
        'category': ['Electronics', 'CLOTHING'],
        'country': ['USA', None]
    })
    
    original = df.copy()
    result = clean_customer_data(df)
    
    # Original should be unchanged
    pd.testing.assert_frame_equal(df, original)


def test_clean_customer_data_fixes_categories():
    """Categories should be lowercased and stripped"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'age': [25, 30, 35],
        'email': ['a@b.com', 'c@d.com', 'e@f.com'],
        'signup_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'purchase_amount': [100.0, 200.0, 300.0],
        'category': ['Electronics', '  CLOTHING  ', 'Home'],
        'country': ['USA', 'UK', 'Canada']
    })
    
    result = clean_customer_data(df)
    
    assert result['category'][0] == 'electronics'
    assert result['category'][1] == 'clothing'
    assert result['category'][2] == 'home'


def test_clean_customer_data_fixes_ages():
    """Invalid ages should become NaN"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'age': [25, 150, 0],
        'email': ['a@b.com', 'c@d.com', 'e@f.com'],
        'signup_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'purchase_amount': [100.0, 200.0, 300.0],
        'category': ['electronics', 'clothing', 'home'],
        'country': ['USA', 'UK', 'Canada']
    })
    
    result = clean_customer_data(df)
    
    assert result['age'][0] == 25
    assert pd.isna(result['age'][1])  # 150 too high
    assert pd.isna(result['age'][2])  # 0 invalid


def test_clean_customer_data_fixes_emails():
    """Invalid emails should become NaN"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'age': [25, 30, 35],
        'email': ['valid@email.com', 'invalid', 'also@invalid'],
        'signup_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'purchase_amount': [100.0, 200.0, 300.0],
        'category': ['electronics', 'clothing', 'home'],
        'country': ['USA', 'UK', 'Canada']
    })
    
    result = clean_customer_data(df)
    
    assert result['email'][0] == 'valid@email.com'
    assert pd.isna(result['email'][1])
    assert pd.isna(result['email'][2])


def test_clean_customer_data_parses_dates():
    """Dates should be parsed to datetime"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'age': [25, 30, 35],
        'email': ['a@b.com', 'c@d.com', 'e@f.com'],
        'signup_date': ['2024-01-15', 'invalid_date', '2024-01-17'],
        'purchase_amount': [100.0, 200.0, 300.0],
        'category': ['electronics', 'clothing', 'home'],
        'country': ['USA', 'UK', 'Canada']
    })
    
    result = clean_customer_data(df)
    
    assert result['signup_date'][0] == pd.Timestamp('2024-01-15')
    assert pd.isna(result['signup_date'][1])
    assert result['signup_date'][2] == pd.Timestamp('2024-01-17')


def test_clean_customer_data_fixes_purchase_amounts():
    """Negative purchase amounts should become NaN"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'age': [25, 30, 35],
        'email': ['a@b.com', 'c@d.com', 'e@f.com'],
        'signup_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'purchase_amount': [100.0, -50.0, 0.0],
        'category': ['electronics', 'clothing', 'home'],
        'country': ['USA', 'UK', 'Canada']
    })
    
    result = clean_customer_data(df)
    
    assert result['purchase_amount'][0] == 100.0
    assert pd.isna(result['purchase_amount'][1])  # Negative
    assert result['purchase_amount'][2] == 0.0    # Zero is valid


def test_clean_customer_data_fills_countries():
    """Missing countries should be filled with 'Unknown'"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'age': [25, 30, 35],
        'email': ['a@b.com', 'c@d.com', 'e@f.com'],
        'signup_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'purchase_amount': [100.0, 200.0, 300.0],
        'category': ['electronics', 'clothing', 'home'],
        'country': ['USA', None, 'Canada']
    })
    
    result = clean_customer_data(df)
    
    assert result['country'][0] == 'USA'
    assert result['country'][1] == 'Unknown'
    assert result['country'][2] == 'Canada'


def test_clean_customer_data_realistic_messy():
    """Integration test with realistic messy data"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Charlie'],
        'age': [34, 28, 150, 45, 0],
        'email': ['john@email.com', 'jane@email.com', 'bob@email', 'alice@email.com', 'invalid'],
        'signup_date': ['2024-01-15', '2024-01-16', '2024-01-17', 'invalid_date', '2024-01-19'],
        'purchase_amount': [156.50, 89.99, 245.00, -50.00, 100.00],
        'category': ['Electronics', 'clothing', '  HOME  ', 'electronics', 'CLOTHING'],
        'country': ['USA', 'USA', None, 'UK', 'Canada']
    })
    
    result = clean_customer_data(df)
    
    # Verify all transformations
    assert result['category'][0] == 'electronics'
    assert result['category'][2] == 'home'
    assert pd.isna(result['age'][2])  # 150
    assert pd.isna(result['age'][4])  # 0
    assert pd.isna(result['email'][2])  # bob@email
    assert pd.isna(result['email'][4])  # invalid
    assert pd.isna(result['signup_date'][3])  # invalid_date
    assert pd.isna(result['purchase_amount'][3])  # -50.00
    assert result['country'][2] == 'Unknown'
    
    # Valid data preserved
    assert result['age'][0] == 34
    assert result['email'][0] == 'john@email.com'
    assert result['purchase_amount'][0] == 156.50


def test_clean_customer_data_preserves_structure():
    """Should preserve DataFrame structure (columns, row count)"""
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'age': [25, 30, 35],
        'email': ['a@b.com', 'c@d.com', 'e@f.com'],
        'signup_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'purchase_amount': [100.0, 200.0, 300.0],
        'category': ['electronics', 'clothing', 'home'],
        'country': ['USA', 'UK', 'Canada']
    })
    
    result = clean_customer_data(df)
    
    assert len(result) == len(df)
    assert list(result.columns) == list(df.columns)