import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
monthly_revenue_df = monthly_revenue.reset_index()
monthly_revenue_df['InvoiceDate'] = monthly_revenue_df['InvoiceDate'].dt.to_timestamp()
monthly_revenue_df.columns = ['Month','Revenue']




yearly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('Y'))['Revenue'].sum()
yearly_revenue_df = yearly_revenue.reset_index()
yearly_revenue_df['InvoiceDate'] = yearly_revenue_df['InvoiceDate'].dt.to_timestamp()
yearly_revenue_df.columns = ['Year','Revenue']


top_10_products = df.groupby('Description')['Quantity'].sum().nlargest(10)
top_10_products_df = top_10_products.reset_index()


top_10_countries = df.groupby('Country')['Quantity'].sum().nlargest(10)
top_10_countries_df = top_10_countries.reset_index()


avg_basket_size = df.groupby(df['InvoiceDate'].dt.to_period('M'))['Quantity'].mean()
avg_basket_size_df = avg_basket_size.reset_index()
avg_basket_size_df['InvoiceDate'] = avg_basket_size_df['InvoiceDate'].dt.to_timestamp()
avg_basket_size_df.columns = ['Month','Basket_size']


customer_spend = df.groupby('CustomerID')['Revenue'].sum()

# Top/Bottom 5%
percent = total_customers * 5 // 100
top_5_percent = customer_spend.nlargest(percent)
bottom_5_percent = customer_spend.nsmallest(percent)

top_5_percent_df = top_5_percent.reset_index()
bottom_5_percent_df = bottom_5_percent.reset_index()


# Repeat vs one-time customers
repeat_customers = (df.groupby('CustomerID')['InvoiceNo'].nunique() > 1).sum()

one_time = (df.groupby('CustomerID')['InvoiceNo'].nunique() == 1).sum()

percent_repeat = (repeat_customers / total_customers) * 100
percent_onetime = 100 - percent_repeat

# Average price and top revenue products
avg_price = df['UnitPrice'].mean()

top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(5)
top_revenue_products_df = top_revenue_products.reset_index()

# RFM Analysis
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Revenue': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']


# Print some summaries
print("Total Transactions:", total_transactions)
print("Total Customers:", total_customers)
print("Overall Revenue:", overall_revenue)
print("Top Products:", top_10_products)
print("Top Countries:", top_10_countries)
print("Top 5% Customers:", top_5_percent)
print("Bottom 5% Customers:", bottom_5_percent)
print("RFM Head:\n", rfm.head())

df.to_csv("Cleaned-Ecommerce-Dataset-Visualization.csv",index=False)
rfm.to_csv("rfm_analysis.csv",index=False)


plt.figure()
sns.lineplot(x = 'Month',y='Revenue', data=monthly_revenue_df)
plt.title("Monthly Revenue")
plt.savefig('visuals/monthly_revenue.png')


plt.figure()
sns.lineplot(x= 'Month',y='Basket_size',data=avg_basket_size_df,palette='rainbow')
plt.title('Average basket size per month')
plt.savefig('visuals/avg_basket_size.png')

plt.figure()
sns.barplot(x = yearly_revenue_df['Year'],y = yearly_revenue_df['Revenue'],palette='deep')
plt.title("Yearly Revenue")
plt.savefig('visuals/yearly_revnue.png')

plt.figure()
sns.barplot(x = top_10_products_df['Quantity'],y=top_10_products_df['Description'])
plt.title("Top 10 Products sold")
plt.savefig('visuals/top_10_product_sold.png')

plt.figure()
sns.barplot(x=top_revenue_products_df['Revenue'],y=top_revenue_products_df['Description'],palette='rainbow')
plt.title("Top 5 revenue Products")
plt.savefig("visuals/Top 5 Revenue products.png")

plt.figure()
sns.histplot(df['UnitPrice'],bins=50,kde=True)
plt.title('Average Unit Price')

plt.figure()
sns.barplot(x=top_10_countries_df['Quantity'],y=top_10_countries_df['Country'],palette='dark')
plt.title('Top 10 countries with most prodcuts delivered')
plt.savefig('visuals/Top 10 countries with most products delivered.png')

plt.figure()
sns.barplot(x = top_5_percent_df['Revenue'],y=top_5_percent_df['CustomerID'],palette='pastel')
plt.title("Top 5% Customer Spent")

plt.figure()
sns.barplot(x = bottom_5_percent_df['Revenue'],y=bottom_5_percent_df['CustomerID'],palette='pastel')
plt.title("Lowest 5% Customer Spent")

plt.figure()
sns.barplot(x = ['One Timer Buyers','Repeated Buyers'] , y =[one_time,repeat_customers])
plt.savefig('visuals/One_Time vs Reapeated Customers.png')

plt.figure()
sns.scatterplot(data = rfm,x = 'Recency',y='Monetary',hue='Frequency',palette='viridis')

plt.figure()
sns.heatmap(df[['Quantity','UnitPrice','Revenue']].corr(),annot=True,cmap='coolwarm')


plt.figure()
sns.boxplot(x=df['Revenue'])


plt.show()