"""
Renewable Energy Data Preprocessing
Prepares data for country comparison dashboard
"""

import pandas as pd
import numpy as np

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw renewable energy data and perform initial cleaning.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Cleaned DataFrame with proper data types
    """
    df = pd.read_csv(filepath)
    
    # Convert TIME to datetime
    df['date'] = pd.to_datetime(df['TIME'], format='%B %Y')
    
    # Ensure numeric columns are correct type
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df['share'] = pd.to_numeric(df['share'], errors='coerce')
    
    return df


def filter_to_renewables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to only renewable energy sources.
    Excludes aggregates and fossil fuels.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Filtered dataframe with only renewable sources
    """
    # Define renewable energy products
    renewable_products = [
        'Hydro',
        'Wind',
        'Solar',
        'Geothermal',
        'Other renewables',
        'Biofuels'
    ]
    
    # Filter to only these products
    df_renewables = df[df['PRODUCT'].isin(renewable_products)].copy()
    
    print(f"Original rows: {len(df)}")
    print(f"Renewable-only rows: {len(df_renewables)}")
    print(f"Products in filtered data: {df_renewables['PRODUCT'].unique()}")
    
    return df_renewables


def calculate_country_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate key metrics for each country:
    - Total renewable production (2010-2022)
    - Average renewable share
    - Growth rate (2022 vs 2010)
    - Number of renewable types used
    
    Args:
        df: Filtered renewable-only dataframe
        
    Returns:
        DataFrame with one row per country and calculated metrics
    """
    metrics = []
    
    for country in df['COUNTRY'].unique():
        country_data = df[df['COUNTRY'] == country]
        
        # Total renewable production across all years
        total_production = country_data['VALUE'].sum()
        
        # Average renewable share
        avg_share = country_data['share'].mean()
        
        # Growth rate: compare 2022 to 2010
        production_2010 = country_data[country_data['YEAR'] == 2010]['VALUE'].sum()
        production_2022 = country_data[country_data['YEAR'] == 2022]['VALUE'].sum()
        
        if production_2010 > 0:
            growth_rate = ((production_2022 - production_2010) / production_2010) * 100
        else:
            growth_rate = np.nan
        
        # Diversity: how many different renewable types?
        num_renewable_types = country_data['PRODUCT'].nunique()
        
        metrics.append({
            'country': country,
            'total_production': total_production,
            'avg_renewable_share': avg_share,
            'growth_rate_pct': growth_rate,
            'num_renewable_types': num_renewable_types,
            'production_2010': production_2010,
            'production_2022': production_2022
        })
    
    return pd.DataFrame(metrics)


def get_time_series_by_country(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Get monthly time series for top N countries by total production.
    
    Args:
        df: Filtered renewable-only dataframe
        top_n: Number of top countries to include
        
    Returns:
        DataFrame with date, country, and total renewable production
    """
    # Get top countries by total production
    country_totals = df.groupby('COUNTRY')['VALUE'].sum().sort_values(ascending=False)
    top_countries = country_totals.head(top_n).index.tolist()
    
    # Filter to top countries
    df_top = df[df['COUNTRY'].isin(top_countries)].copy()
    
    # Aggregate by country and date (sum all renewable types per month)
    time_series = df_top.groupby(['COUNTRY', 'date'])['VALUE'].sum().reset_index()
    time_series.columns = ['country', 'date', 'total_renewable_production']
    
    return time_series


def identify_storage_candidates(country_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a "storage need score" based on:
    - High renewable share (more renewables = more variability)
    - High growth rate (rapid growth = infrastructure strain)
    - Multiple renewable types (diversity = complexity)
    
    Args:
        country_metrics: DataFrame from calculate_country_metrics()
        
    Returns:
        DataFrame with storage_score column added, sorted by score
    """
    df = country_metrics.copy()
    
    # Normalize metrics to 0-1 scale for scoring
    df['share_normalized'] = (df['avg_renewable_share'] - df['avg_renewable_share'].min()) / \
                              (df['avg_renewable_share'].max() - df['avg_renewable_share'].min())
    
    df['growth_normalized'] = (df['growth_rate_pct'] - df['growth_rate_pct'].min()) / \
                               (df['growth_rate_pct'].max() - df['growth_rate_pct'].min())
    
    df['diversity_normalized'] = (df['num_renewable_types'] - df['num_renewable_types'].min()) / \
                                  (df['num_renewable_types'].max() - df['num_renewable_types'].min())
    
    # Calculate storage score (weighted sum)
    df['storage_score'] = (
        0.4 * df['share_normalized'] +      # 40% weight: current renewable penetration
        0.4 * df['growth_normalized'] +     # 40% weight: growth trajectory
        0.2 * df['diversity_normalized']    # 20% weight: renewable mix complexity
    ) * 100  # Scale to 0-100
    
    # Sort by storage score
    df = df.sort_values('storage_score', ascending=False)
    
    return df


def main():
    """
    Run complete preprocessing pipeline and save results.
    """
    print("Loading data...")
    df = load_and_clean_data('data/renewable_energy.csv')
    
    print("\nFiltering to renewable sources...")
    df_renewables = filter_to_renewables(df)
    
    print("\nCalculating country metrics...")
    country_metrics = calculate_country_metrics(df_renewables)
    
    print("\nExtracting time series for top countries...")
    time_series = get_time_series_by_country(df_renewables, top_n=15)
    
    print("\nIdentifying storage investment candidates...")
    storage_candidates = identify_storage_candidates(country_metrics)
    
    # Save processed data
    df_renewables.to_csv('data_clean_renewables.csv', index=False)
    country_metrics.to_csv('data_country_metrics.csv', index=False)
    time_series.to_csv('data_time_series.csv', index=False)
    storage_candidates.to_csv('data_storage_candidates.csv', index=False)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nTop 5 Storage Investment Candidates:")
    print(storage_candidates[['country', 'avg_renewable_share', 'growth_rate_pct', 'storage_score']].head())
    
    print(f"\nFiles saved:")
    print("  - data_clean_renewables.csv (filtered dataset)")
    print("  - data_country_metrics.csv (aggregated country stats)")
    print("  - data_time_series.csv (monthly trends for top countries)")
    print("  - data_storage_candidates.csv (ranked by storage need)")


if __name__ == "__main__":
    main()