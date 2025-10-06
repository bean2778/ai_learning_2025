import pandas as pd

def get_customer_summary(transactions_df) -> pd.DataFrame:
    """
    Per customer:
        total amount spent
        Number of transactions
        Average Transaction amount
        Most Purchased Cat
    For most transactions, I'm not taking quantity into account
    """
    total_spent = transactions_df.groupby('customer_id')['amount'].sum()
    average_trans = transactions_df.groupby('customer_id')['amount'].mean()
    num_trans = transactions_df.groupby('customer_id')['transaction_id'].count()
    cat_counts = transactions_df.groupby(['customer_id', 'product_category']).size()
    most_cat = cat_counts.groupby('customer_id').idxmax()
    summary = pd.DataFrame({
        'total_spent' : total_spent,
        'num_transactions' : num_trans,
        'avg_transaction' : average_trans,
        'most_purchased_category' : most_cat
    })
    return summary

def get_country_analysis(customers_df, transactions_df) -> pd.DataFrame:
    """
    Total revenue
    Number of customers
    Average revenue per customer
    number of transactions
    """
    merged = pd.merge(customers_df, transactions_df, on='customer_id', how='inner')
    total_revenue = merged.groupby('country')['amount'].sum()
    num_customers = merged.groupby('country')['customer_id'].nunique()
    average_rev_per_customer = total_revenue / num_customers
    num_trans = merged.groupby('country')['transaction_id'].nunique()
    print(num_trans)
    summary = pd.DataFrame({
        'total_revenue' : total_revenue,
        'number_customers' : num_customers,
        'average_rev_per_customer' : average_rev_per_customer,
        'number_transactions' : num_trans
    })
    return summary

def get_monthly_trends(transactions_df) -> pd.DataFrame:
    """
    Per month:
        total revenue
        number of transactions
        average transaction amount
    """
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

    transactions_df['year_month'] = transactions_df['transaction_date'].dt.to_period('M')
    monthly = transactions_df.groupby('year_month').agg({
        'amount': ['sum', 'count', 'mean']
    })
    return monthly

def calculate_customer_lifetime_value(customers_df, transactions_df) -> pd.DataFrame:
    merged = pd.merge(customers_df, transactions_df, on='customer_id', how='inner')
    merged['transaction_date'] = pd.to_datetime(merged['transaction_date'])
    names = merged.groupby('customer_id')['name'].first()
    countries = merged.groupby('customer_id')['country'].first()
    segment = merged.groupby('customer_id')['customer_segment'].first()
    total_spent = merged.groupby('customer_id')['amount'].sum()
    long_ago = merged.groupby('customer_id')['transaction_date'].min()
    short_ago = merged.groupby('customer_id')['transaction_date'].max()
    days_since_first = (pd.Timestamp.now() - long_ago).dt.days
    days_since_last = (pd.Timestamp.now() - short_ago).dt.days

    summary = pd.DataFrame({
        'name' : names,
        'country': countries,
        'segment' : segment,
        'total_spent' : total_spent,
        'days_since_first' : days_since_first,
        'days_since_last' : days_since_last
    })
    return summary


customers = pd.read_csv('customers.csv')
transactions = pd.read_csv('transactions.csv')
# print(customers.head())
# print(customers.info())

# print(transactions.head())
# print(transactions.info())

# print(transactions.groupby('customer_id')['amount'].sum())

# merged = pd.merge(customers, transactions, on='customer_id', how='inner')
# print(merged.head)

# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# get_customer_summary(transactions)
print(calculate_customer_lifetime_value(customers, transactions))