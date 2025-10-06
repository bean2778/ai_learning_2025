import pandas as pd
import numpy as np


def clean_email_addresses(series: pd.Series) -> pd.Series:
    """
    Fix common email format issues.
    
    Rules to implement:
    - Valid email must have @ and a domain
    - Mark invalid emails as None/NaN
    """
    cleaned = series.copy()
    pattern = r'[\w.]+@\w+\..+'
    is_valid = series.str.contains(pattern, regex=True, na=False)
    cleaned[~is_valid] = np.nan
    return cleaned

def standardize_categories(series: pd.Series) -> pd.Series:
    """
    Standardize category names (lowercase, consistent format).
    """
    cleaned = series.copy()
    cleaned = cleaned.str.strip()      
    cleaned = cleaned.str.lower()
    cleaned = cleaned.mask(cleaned == '', np.nan)
    return cleaned

def clean_age_values(series: pd.Series) -> pd.Series:
    """
    Handle impossible ages.
    
    Rules:
    - Ages must be between 1 and 120
    - Replace invalid with NaN
    """
    return series.mask((series > 120) | (series < 1))

def clean_dates(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    return pd.to_datetime(cleaned, errors='coerce')

def clean_purchase_amounts(series: pd.Series) -> pd.Series:
    """
    Clean purchase amounts.
    
    Rules:
    - Negative amounts become NaN
    - Zero is technically valid (free item/return)
    """
    return series.mask(series < 0)

def clean_country(series: pd.Series) -> pd.Series:
    return series.fillna('Unknown')

def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean['email'] = clean_email_addresses(df_clean['email'])
    df_clean['category'] = standardize_categories(df_clean['category'])
    df_clean['age'] = clean_age_values(df_clean['age'])
    df_clean['signup_date'] = clean_dates(df_clean['signup_date'])
    df_clean['purchase_amount'] = clean_purchase_amounts(df_clean['purchase_amount'])
    df_clean['country'] = clean_country(df_clean['country'])
    return df_clean



def main():
    df = pd.read_csv('messy_customer_data.csv')
    print(df)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(clean_customer_data(df))

        
if __name__ == "__main__":
    main()