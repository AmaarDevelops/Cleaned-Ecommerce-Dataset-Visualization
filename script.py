import pandas as pd
import numpy as np
import os

# Get the folder where the script is running
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Online_Retail.csv')
df = pd.read_csv(file_path)

df.drop_duplicates(inplace=True)

# Fill missing customer IDs
df['CustomerID'].fillna('Guest', inplace=True)

# Convert invoice date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Feature: Revenue
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Total statistics
total_transactions = df['InvoiceNo'].nunique()
total_customers = df['CustomerID'].nunique()
overall_revenue = df['Revenue'].sum()

# Grouping
monthly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('M'))['Revenue'].sum()
yearly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('Y'))['Revenue'].sum()
top_10_products = df.groupby('Description')['Quantity'].sum().nlargest(10)
top_10_countries = df.groupby('Country')['Quantity'].sum().nlargest(10)

avg_basket_size = df.groupby(df['InvoiceDate'].dt.to_period('M'))['Quantity'].mean()
customer_spend = df.groupby('CustomerID')['Revenue'].sum()

# Top/Bottom 5%
percent = total_customers * 5 // 100
top_5_percent = customer_spend.nlargest(percent)
bottom_5_percent = customer_spend.nsmallest(percent)

# Repeat vs one-time customers
repeat_customers = (df.groupby('CustomerID')['InvoiceNo'].nunique() > 1).sum()
percent_repeat = (repeat_customers / total_customers) * 100
percent_onetime = 100 - percent_repeat

# Average price and top revenue products
avg_price = df['UnitPrice'].mean()
top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(5)

# RFM Analysis
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Revenue': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Save outputs
df.to_csv('cleaned_ecommerce_dataset.csv', index=False)
rfm.to_csv('rfm_analysis.csv')

# Print some summaries
print("Total Transactions:", total_transactions)
print("Total Customers:", total_customers)
print("Overall Revenue:", overall_revenue)
print("Top Products:", top_10_products)
print("Top Countries:", top_10_countries)
print("Top 5% Customers:", top_5_percent)
print("Bottom 5% Customers:", bottom_5_percent)
print("RFM Head:\n", rfm.head())
