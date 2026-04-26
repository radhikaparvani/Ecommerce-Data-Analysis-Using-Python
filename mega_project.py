import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''Section 1: Data Loading and Inspection
•	Load all datasets into Pandas DataFrames
•	Display the first 10 rows of each dataset
•	Check dataset shape (rows and columns)
•	Inspect column data types
•	Identify missing values
'''

#----------------1.loading all Datasets-------------------
customers = pd.read_csv("data/mega_customers.csv")
items = pd.read_csv("data/mega_order_items.csv")
orders = pd.read_csv("data/mega_orders.csv")
products = pd.read_csv("data/mega_products.csv")

#----------------2.Display first 10 rows-------------------
print(customers.head(10))
print(items.head(10))
print(orders.head(10))
print(products.head(10))

#----------------3.Check Data Shape(rows & columns)--------
print(customers.shape)
print(items.shape)
print(orders.shape)
print(products.shape)

#-----------------4.Column Data Type-----------------------
print(customers.dtypes)
print(items.dtypes)
print(orders.dtypes)
print(products.dtypes)

#------------------5.identifying missing values-------------
print(customers.isnull().sum())
print(items.isnull().sum())
print(orders.isnull().sum())
print(products.isnull().sum())

'''Section 2: Data Cleaning
•	Handle missing values using appropriate techniques
•	Remove duplicate records
•	Correct inconsistent or incorrect data types
•	Handle invalid or corrupted entries
•	Standardize column names
'''

print("starting data cleaning...\n")
#-----------1.Handling Missing values-------------------
customers.dropna(inplace=True)
items.dropna(inplace=True)
orders.dropna(inplace=True)
products.dropna(inplace=True)

#-----------2.Remove duplicate records------------------
customers.drop_duplicates(inplace=True)
items.drop_duplicates(inplace=True)
orders.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)

#-----------3.fix DataTypes----------------------------
customers["signup_date"]=pd.to_datetime(customers["signup_date"],errors="coerce")
orders["order_date"]=pd.to_datetime(orders["order_date"],errors="coerce")

#-----------4.Handle Invalid Entries-------------------
print("Invalid Signup Date: ",customers["signup_date"].isnull().sum())
print("Invalid Order Date: ",orders["order_date"].isnull().sum())

#-----------5.Standardize column names-----------------
customers.columns = customers.columns.str.lower().str.strip().str.replace(" ","_")#here strip is used to remove extra space
items.columns = items.columns.str.lower().str.strip().str.replace(" ","_")
orders.columns = orders.columns.str.lower().str.strip().str.replace(" ","_")
products.columns = products.columns.str.lower().str.strip().str.replace(" ","_")

'''Section 3: Data Integration
•	Merge customers and orders datasets
•	Merge products and order_items datasets
•	Create a final master dataset
•	Validate joins to ensure no data loss
•	Verify row consistency after merging
'''

#----------------1.Merging Customer-Order DataSet--------------
customer_orders = pd.merge(customers, orders, on="customer_id", how="inner")
print("customers and orders dataset merged successfully.\n")
print(customer_orders.head(10),'\n')

#----------------2.Merging Products and order_items dataset-----
product_items= pd.merge(products,items, on="product_id", how="inner")
print("products and items dataset merged successfully.\n")
print(product_items.head(10),'\n')

#---------------3.Creating Master Dataset-------------------
final_data = pd.merge(customer_orders, product_items, on="order_id", how="inner")
print("final master dataset created successfully.\n")
print(final_data.head(10),"\n")

#--------------4.Ensuring no data loss--------------------
print("checking key columns for missing values after merge...\n")
print("customer-orders customer_id missing:",customer_orders["customer_id"].isnull().sum())
print("customer-ordes order_id missing:",customer_orders["order_id"].isnull().sum())
print("product-items product_id missing",product_items["product_id"].isnull().sum())
print("product-items order_id missing",product_items["order_id"].isnull().sum())
print("final data order_id missing:",final_data["order_id"].isnull().sum())

#--------------5.Verifying row consistency---------------
print("original customers rows:",customers.shape[0])
print("original items rows:",items.shape[0])
print("original orders rows:",orders.shape[0])
print("original products rows:",products.shape[0])

print("merged customer-ordes rows:",customer_orders.shape[0])
print("merged products-items rows:",product_items.shape[0])
print("final dataset rows:",final_data.shape[0])

#--------------6.Displaying Columns of final dataset--------
print("Final Dateset Columns:",final_data.columns)

'''Section 4: Feature Engineering
•	Create total_order_value column
•	Extract month and year from date columns
•	Create customer_age_group categories
•	Generate high_value_flag column
•	Create repeat_customer_flag
'''
print("starting feature engineering...\n")
#---------1.Create Total order value multiply by price and quantity----
final_data["total_order_value"]=final_data["price"]*final_data["quantity"]
print("total order value column created successfully.\n")

#---------2.extracting Month and Year From OrderDate--------
final_data["order_month"]=final_data["order_date"].dt.month
final_data["order_year"]=final_data["order_date"].dt.year
print("Month and Year column created successfully.\n")

#--------3.Create Customer Age Group----------------------
final_data["customer_age_group"]=pd.cut(
    final_data["age"],
    bins=[0,18,35,50,100],
    labels=["Child","Adult","Middle Age","Senior"]
)
print("customer_age_group column created successfully.\n")

#-------4.create high Value Flags-------------------------
final_data["high_value_flag"]=np.where(final_data["total_order_value"]>1000,1,0)
print("high value flag column created successfully.\n")

#------5.Create Repeat Customer flag----------------------
repeat_customers = final_data.groupby("customer_id")["order_id"].nunique()
final_data["repeat_customer_flag"]=final_data["customer_id"].map(
    lambda x:1 if repeat_customers[x] > 1 else 0
)
print("repeat_customers column created successfully.\n")

#------6.Preview Updated Dateset-----------------
print("Feature Engineering Complete!\n")
print(final_data.head(10))

'''Section 5: Basic Statistics (NumPy)
•	Calculate mean order value
•	Calculate median order value
•	Compute standard deviation
•	Identify maximum and minimum order values
•	Perform percentile analysis
'''

print("starting Basic Statistics Anaylsis...\n")

#-------------1.Extract Order Values-------------
order_values = final_data["total_order_value"].values

#-------------2.Mean----------------------------
mean = np.mean(order_values)
print("mean order value:", mean)

#-------------3.Median--------------------------
median = np.median(order_values)
print("median order value:", median)

#-------------4.Standard Deviation---------------
standard_deviation = np.std(order_values)
print("standard deviation:", standard_deviation)

#--------------5.Max&Min------------------------
max_values = np.max(order_values)
print("max order value:", max_values)
min_values = np.min(order_values)
print("min order value:", min_values)

#------------6.Percentiles----------------------
p25 = np.percentile(order_values, 25)
p50 = np.percentile(order_values, 50)
p75 = np.percentile(order_values, 75)
print("25th percentile:", p25)
print("50th percentile:", p50)
print("75th percentile:", p75)

print("Basic Analysis Complete!\n")

'''Section 6: Sales Analysis
•	Calculate total revenue
•	Analyze revenue by month
•	Analyze revenue by product category
•	Identify daily sales trends
•	Determine peak sales day
'''
print("Starting Sales Anaylsis...\n")

#----------1.Total Revenue--------------------------
total_revenue = final_data["total_order_value"].sum()
print("total revenue:", total_revenue,"\n")

#---------2.Monthly Revenue-------------------------
monthly_revenue = final_data.groupby("order_month")["total_order_value"].sum()
print("monthly revenue:", monthly_revenue,'\n')

#---------3.Category Revenue-----------------------
category_revenue = final_data.groupby("category")["total_order_value"].sum()
print("category revenue:", category_revenue)

#----------4.Daily Sales---------------------------
daily_sales = final_data.groupby("order_date")["total_order_value"].sum()
print("daily sales:", daily_sales)

#----------5.Peak Sales day----------------------
peak_day = daily_sales.idxmax()
print("peak day:", peak_day)
peak_value =daily_sales.max()
print("peak value:", peak_value)

print("Sales Anaylsis Complete!\n")

'''Section 7: Customer Analysis
•	Count total number of customers
•	Identify new versus returning customers
•	Calculate average orders per customer
•	Identify top 10 customers
•	Calculate customer contribution percentage
'''
print("Starting Customer Anaylsis...\n")

#-------------1.total Customers--------------------
total_customers = final_data["customer_id"].nunique()
print("total customers:", total_customers,'\n')

#------------2.New & Returning Customers-----------
customer_orders = final_data.groupby("customer_id")["order_id"].nunique()
new_customers = (customer_orders==1).sum()
print("new customers:", new_customers)
returning_customers = (customer_orders >  1).sum()
print("returning customers:", returning_customers,'\n')

#-----------3.Avg Order Per Customer-------------
total_orders = final_data["order_id"].nunique()
total_customers = final_data["customer_id"].nunique()
avg_orders = total_orders/total_customers
print("average orders:", avg_orders,'\n')

#-----------4.Top 10 Customers-------------------
customer_revenue = final_data.groupby("customer_id")["total_order_value"].sum()
top_customers = customer_revenue.sort_values(ascending=False).head(10)
print("top customers:", top_customers,'\n')

#-----------5.Customer Contribution Percentage------
customer_contribution = (customer_revenue/total_revenue)*100
print("customer contribution:", customer_contribution,'\n')

print("Customer Anaylsis Complete!\n")

'''Section 8: Customer Segmentation
•	Create High, Medium, and Low value segments
•	Segment customers based on spending
•	Segment customers based on purchase frequency
•	Combine segmentation logic
•	Visualize customer segments
'''
print("Starting Customer Segmentation...\n")

#------------1.Segment Based on Customer Spending------------
customer_spending = final_data.groupby("customer_id")["total_order_value"].sum().reset_index()
customer_spending["spending_segment"]=pd.cut(
    customer_spending["total_order_value"],
    bins=[0,15000,50000,float("inf")],
    labels=["Low","Medium","High"]
)
print("Customer Spending Segment:\n")
print(customer_spending.head(10),'\n')

#-------2.Segment Based on Customer Purchase Frequency---------
customer_orders = final_data.groupby("customer_id")["order_id"].nunique().reset_index()
customer_orders["Frequency_segment"]=pd.cut(
    customer_orders["order_id"],
    bins=[0,5,50,float("inf")],
    labels=["Low","Medium","High"]
)
print("customer frequency segment:\n")
print(customer_orders.head(10),"\n")

#----------3.combine segmentation logic--------------
customer_segment = pd.merge(
    customer_spending,
    customer_orders,
    on="customer_id",
    how="inner"
)
print(customer_segment.head(10))

#----------4.create final customer segment-------------
def final_segment(row):
    if row["spending_segment"]=="High" and row["Frequency_segment"]=="High":
        return "Premium"
    elif row["spending_segment"]=="Medium" and row["Frequency_segment"]=="Medium":
        return "Regular"
    else:
        return "Low Value"

customer_segment["final_segment"]=customer_segment.apply(final_segment, axis=1)
print(customer_segment.head(10))

#-----------5.Visualize customer segments----------------
customer_segment["final_segment"].value_counts().plot(kind="bar")
plt.title("Custom Segments distribution")
plt.xlabel(" Segment")
plt.ylabel("number of customers")
plt.show()

'''Section 9: Product Performance
•	Identify top-selling products
•	Identify low-performing products
•	Analyze category-wise sales
•	Calculate average price per category
•	Determine product contribution to revenue
'''

print("Starting Product Performance anaylsis...\n")

#----------1.Top selling products---------------
product_sales = final_data.groupby("product_id")["total_order_value"].sum()
top_products = product_sales.sort_values(ascending=False).head(10)
print("top 10  products:", top_products,'\n')

#---------2.Low Performing products-------------
low_products = product_sales.sort_values(ascending=True).head(10)
print("low products:", low_products)

#--------3.Catogery Wise Sales------------------
category_sales = final_data.groupby("category")["total_order_value"].sum()
print("category sales:", category_sales.sort_values(ascending=False))

#---------4.Visualization-----------------------
category_sales.plot(kind="bar",figsize=(8,5))
plt.title("Category Sales Distribution")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.show()

#-----------5.avg price per category---------------
avg_price = final_data.groupby("category")["price"].mean()
print("average price per category:", avg_price.sort_values(ascending=False))

#------------6.Product contribution to revenue-------
total_revenue = final_data["total_order_value"].sum()

product_revenue = final_data.groupby("product_id")["total_order_value"].sum()

product_contribution = (product_revenue/total_revenue)*100
print("product contribution:", product_contribution.head())

'''Section 10: Time Series Analysis
•	Analyze daily sales trends
•	Analyze weekly sales patterns
•	Analyze monthly trends
•	Identify seasonality
•	Calculate growth rate
'''
print("Starting Time Series Analysis...\n")

#----------1.Daily Sales Trends-------------
daily_sales = final_data.groupby("order_date")["total_order_value"].sum()

print("daily sales:", daily_sales)
daily_sales.plot(kind="bar",figsize=(8,5))
plt.title("Daily Sales Trends")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.xticks(rotation=45)
plt.show()

#----------2.Weekly Sales------------------
final_data["week"]=final_data["order_date"].dt.to_period("W")
weekly_sales = final_data.groupby("week")["total_order_value"].sum()
print("weekly sales:", weekly_sales)
weekly_sales.plot(kind="bar",figsize=(8,5))
plt.title("Weekly Sales Trends")
plt.xlabel("Week")
plt.ylabel("weekly Sales")
plt.xticks(rotation=45)
plt.show()

#-----------3.Monthly sales---------------
final_data["month"]=final_data["order_date"].dt.to_period("M")
monthly_sales = final_data.groupby("month")["total_order_value"].sum()
print("monthly sales:", monthly_sales)
monthly_sales.plot(kind="bar",figsize=(8,5))
plt.title("Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("monthly Sales")
plt.xticks(rotation=45)
plt.show()

#----------4.Growth Rate------------------
growth_rate = monthly_sales.pct_change() * 100
print("growth rate:", growth_rate)

growth_rate.plot(kind="bar",figsize=(8,5))
plt.title("Growth Rate Trends")
plt.xlabel("month")
plt.ylabel("growth rate")
plt.xticks(rotation=45)
plt.show()

'''Section 11: Moving Average Analysis
•	Calculate 7-day moving average
•	Calculate 30-day moving average
•	Compare trends over time
•	Detect trend direction
•	Smooth noisy data
'''
print("Starting Moving Average Analysis...\n")

#-----------1.calculate moving avg----------------
daily_sales_7 = daily_sales.rolling(window=7).mean()
print("daily sales 7:", daily_sales_7)

daily_sales_30 = daily_sales.rolling(window=30).mean()
print("daily sales 30:", daily_sales_30)

#----------2.comapre actual sales with moving averages---------
daily_sales.plot(figsize=(10,5),label="actual sales")
daily_sales_7.plot(label="7-day average sales")
daily_sales_30.plot(label="30-day average sales")
plt.title("moving average comparison")
plt.xlabel("Date")
plt.ylabel("Moving Average")
plt.legend()
plt.show()

#---------3.Detect Trend direction------------
if daily_sales_7.iloc[-1]> daily_sales_7.iloc[0]:
    print("trend is increasing")
else:
    print("trend is decreasing")

#-------4.Smooth noisy data------------------
plt.figure(figsize=(10,5))
plt.plot(daily_sales,alpha=0.5,label="original(noisy)")
plt.plot(daily_sales_7,alpha=0.5,label="smooth 7 day moving average")
plt.title("noise reduction using moving average")
plt.legend()
plt.tight_layout()
plt.show()

'''Section 12: Advanced Visualization (Matplotlib)
•	Create line charts for sales trends
•	Create bar charts for category analysis
•	Create histograms for distribution analysis
•	Create scatter plots for relationships
•	Build a dashboard using subplots
'''

print("Starting Advanced Visualization...\n")

#----------1.Line chart:Sales Trend------------
daily_sales = final_data.groupby("order_date")["total_order_value"].sum().sort_index()
plt.figure(figsize=(10,5))
plt.plot(daily_sales.index,daily_sales.values)
plt.title("Sales Trends")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

#---------2.Bar Chart:Category Analysis----------
plt.figure(figsize=(10,5))
plt.bar(category_sales.index,category_sales.values)
plt.xticks(rotation=45)
plt.title("Category Sales Distribution")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.show()

#--------3.Histogram : Order Value Distribution---------
plt.figure(figsize=(10,5))
plt.hist(final_data["total_order_value"],bins=20)
plt.title("Order value analysis")
plt.xlabel("Order Value")
plt.ylabel("Frequency")
plt.show()

#---------4.Scatter Plot:Price Vs Quality----------------
plt.figure(figsize=(10,5))
plt.scatter(final_data["price"],final_data["quantity"],alpha=0.5)
plt.title("Price vs Quantity")
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.show()

#-----------5.Dashboard using Subplot--------------------
fig, axs = plt.subplots(2,2,figsize=(14,10))

axs[0,0].plot(daily_sales.index,daily_sales.values)
axs[0,0].set_title("Sales Trend")
axs[0,0].tick_params(axis='x',rotation=45)

axs[0,1].bar(category_sales.index,category_sales.values)
axs[0,1].set_title("Category Trend")
axs[0,1].tick_params(axis='x',rotation=45)

axs[1,0].hist(final_data["total_order_value"],bins=20)
axs[1,0].set_title("Distribution Trend")

axs[1,1].scatter(final_data["price"],final_data["quantity"],alpha=0.5)
axs[1,1].set_title("Price vs Quantity")
axs[1,1].set_xlabel("Price")
axs[1,1].set_ylabel("Quantity")

plt.tight_layout()
plt.show()

'''Section 13: Seaborn Analysis
•	Create heatmap for correlation analysis
•	Use boxplots to identify outliers
•	Use violin plots for distribution
•	Generate pairplots for multi-variable analysis
•	Create countplots for categorical data
'''
print("Starting Seaborn Analysis...\n")

#---------1.Correlation Heatmap------------------
numeric_value = final_data.select_dtypes(include=["int64", "float64"])
corr = numeric_value.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True,cmap="coolwarm")
plt.title("Correlation Analysis")
plt.show()

#-------2.Boxplot For Outliers------------------
plt.figure(figsize=(8,6))
plt.boxplot(x=final_data["total_order_value"])
plt.title("Boxplot of Total Order Value")
plt.xticks(rotation=45)
plt.show()

#--------3.Violin Plot(Distribution)-------------
plt.figure(figsize=(10,6))
plt.violinplot(final_data["total_order_value"])
plt.title("Violin Plot of Total Order Value")
plt.xlabel("Order Value")
plt.tight_layout()
plt.show()

#-------4.Pairplot(multiple analysis)-------------
selected_data = final_data[["price","quantity","total_order_value"]]
sns.pairplot(selected_data)
plt.show()

#-------5.CountPlot Category dist.------------
plt.figure(figsize=(8,6))
sns.countplot(x=final_data["category"])
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

'''Section 14: Correlation Analysis
•	Compute correlation matrix
•	Identify strong relationships
•	Identify weak relationships
•	Interpret correlations
•	Visualize using heatmap
'''
print("Starting Correlation Analysis...\n")

#------------1.Compute Correlation matrix----------
numerical_value = final_data.select_dtypes(include=["int64", "float64"])
corr = numerical_value.corr()
print(corr)

#------------2.Identify Strong Correlation----------
strong_corr = corr[(corr > 0.7) | (corr < -0.7)]
print(strong_corr)

#-----------3.Identify Weak Correlation------------
weak_corr = corr[(corr > -0.2) & (corr < 0.2)]
print(weak_corr)

#-----------4.Visualization------------------
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


'''Section 15: Anomaly Detection
•	Calculate mean and standard deviation
•	Define anomaly threshold
•	Detect outliers
•	Visualize anomalies
•	Analyze abnormal cases
'''

print("Starting Anomaly Detection...\n")

# ---------- 1. Calculate mean & standard deviation ----------
mean_value = final_data["total_order_value"].mean()
std_value = final_data["total_order_value"].std()

print("Mean Value:", mean_value)
print("Standard Deviation:", std_value, "\n")


# ---------- 2. Define anomaly thresholds ----------
upper_limit = mean_value + 2 * std_value
lower_limit = mean_value - 2 * std_value

print("Upper Limit:", upper_limit)
print("Lower Limit:", lower_limit, "\n")


# ---------- 3. Detect anomalies ----------
anomalies = final_data[
    (final_data["total_order_value"] > upper_limit) |
    (final_data["total_order_value"] < lower_limit)
]

normal_data = final_data[
    (final_data["total_order_value"] <= upper_limit) &
    (final_data["total_order_value"] >= lower_limit)
]

print("Total Anomalies Detected:", len(anomalies), "\n")
print("Anomalies Preview:\n", anomalies.head(), "\n")


# ---------- 4. Visualization ----------
plt.figure(figsize=(10, 6))

plt.scatter(normal_data["order_date"], normal_data["total_order_value"],
            alpha=0.4, label="Normal Data")

plt.scatter(anomalies["order_date"], anomalies["total_order_value"],
            color="red", label="Anomalies")

plt.title("Anomaly Detection in Order Values")
plt.xlabel("Date")
plt.ylabel("Order Value")
plt.legend()
plt.tight_layout()
plt.show()


# ---------- 5. Analyze abnormal cases ----------
print("Abnormal Transactions:\n",
      anomalies[["customer_id", "order_id", "total_order_value"]].head(), "\n")

print("Anomaly Detection Completed ")

'''Section 16: Fraud Detection Logic
•	Identify high-value transactions
•	Detect unusual purchase timing
•	Identify multiple rapid orders
•	Create fraud_flag column
•	Analyze fraud patterns
'''

print("Starting Fraud Detection...\n")

# ---------- 1. High-value transactions ----------
high_value_limit = final_data["total_order_value"].mean() + 2 * final_data["total_order_value"].std()

final_data["high_value_txn"] = (final_data["total_order_value"] > high_value_limit).astype(int)


# ---------- 2. Unusual purchase timing ----------
# Assume late night transactions as suspicious
final_data["order_hour"] = final_data["order_date"].dt.hour

final_data["odd_hour_txn"] = final_data["order_hour"].apply(
    lambda x: 1 if x >= 23 or x <= 5 else 0
)


# ---------- 3. Rapid multiple orders ----------
final_data = final_data.sort_values(["customer_id", "order_date"])

final_data["time_diff"] = final_data.groupby("customer_id")["order_date"].diff().dt.total_seconds()

# If orders within 60 seconds → suspicious
final_data["rapid_txn"] = (final_data["time_diff"] < 60).astype(int)


# ---------- 4. Combine fraud logic ----------
final_data["fraud_flag"] = (
    final_data["high_value_txn"] +
    final_data["odd_hour_txn"] +
    final_data["rapid_txn"]
)

# If any condition true → fraud_flag = 1
final_data["fraud_flag"] = (final_data["fraud_flag"] > 0).astype(int)


# ---------- 5. Results ----------
total_fraud = final_data["fraud_flag"].sum()

print("Total Suspicious Transactions:", total_fraud, "\n")

print("Fraudulent Transactions Preview:\n",
      final_data[final_data["fraud_flag"] == 1][
          ["customer_id", "order_id", "product_id",
           "total_order_value", "fraud_flag"]
      ].head(), "\n")

print("Fraud Detection Completed ")

'''Section 17: Behavioral Analysis
•	Analyze purchase frequency
•	Calculate time between orders
•	Identify category preferences
•	Map customer journey
•	Perform engagement analysis
'''

print("Starting Behavioral Analysis...\n")

# ---------- 1. Analyze purchase frequency ----------
customer_freq = final_data.groupby("customer_id")["order_id"].nunique()

print("Customer Purchase Frequency:\n", customer_freq.head(), "\n")


# ---------- 2. Calculate time between orders ----------
final_data = final_data.sort_values(by=["customer_id", "order_date"])

final_data["time_between_orders_days"] = (
    final_data.groupby("customer_id")["order_date"]
    .diff()
    .dt.days
)

print("Time Between Orders:\n",
      final_data[["customer_id", "order_date", "time_between_orders_days"]].head(), "\n")


# ---------- 3. Identify category preferences ----------
category_pref = (
    final_data.groupby(["customer_id", "category"])["order_id"]
    .count()
    .reset_index(name="purchase_count")
)

top_category_pref = category_pref.sort_values(
    ["customer_id", "purchase_count"],
    ascending=[True, False]
).drop_duplicates("customer_id")

print("Top Category Preference:\n", top_category_pref.head(), "\n")


# ---------- 4. Map customer journey ----------
customer_journey = final_data.groupby("customer_id")["category"].apply(list)

print("Customer Journey:\n", customer_journey.head(), "\n")


# ---------- 5. Perform engagement analysis ----------
def engagement(order_count):
    if order_count >= 5:
        return "High"
    elif order_count >= 2:
        return "Medium"
    else:
        return "Low"


engagement_level = customer_freq.apply(engagement)

print("Customer Engagement Level:\n", engagement_level.head(), "\n")

print("Behavioral Analysis Completed ")

'''Section 18: Retention Analysis
•	Classify active versus inactive users
•	Identify customer churn
•	Calculate last purchase gap
•	Determine retention rate
•	Analyze customer lifetime value
'''
print("Starting Retention Analysis...\n")

# ---------- 1. Calculate last purchase gap ----------
max_date = final_data["order_date"].max()
last_purchase = final_data.groupby("customer_id")["order_date"].max()

recency = (max_date - last_purchase).dt.days

print("Customer Recency Preview:\n", recency.head(), "\n")


# ---------- 2. Classify active and inactive users ----------
active_users = (recency <= 30).sum()
inactive_users = (recency > 30).sum()

print("Active Users:", active_users)
print("Inactive Users:", inactive_users, "\n")


# ---------- 3. Identify churn customers ----------
churn_customers = recency[recency > 30]

print("Churn Customers:", len(churn_customers), "\n")


# ---------- 4. Calculate retention rate ----------
total_customers = final_data["customer_id"].nunique()
retention_rate = (active_users / total_customers) * 100

print("Retention Rate:", retention_rate, "%\n")


# ---------- 5. Customer Lifetime Value ----------
customer_lifetime_value = final_data.groupby("customer_id")["total_order_value"].sum()

print("Customer Lifetime Value Preview:\n", customer_lifetime_value.head(), "\n")

print("Retention Analysis Completed ")

'''Section 19: Business Insights
•	Identify most profitable segment
•	Identify worst-performing category
•	Detect high-risk customers
•	Discover growth opportunities
•	Provide strategic recommendations
'''

print("Starting Business Insights...\n")

# ---------- 1. Most profitable segment ----------
revenue_by_segment = customer_segment.groupby("final_segment")["total_order_value"].sum().sort_values(ascending=False)

top_segment = revenue_by_segment.idxmax()

print("Revenue by Segment:\n", revenue_by_segment, "\n")
print("Most Profitable Segment:", top_segment, "\n")


# ---------- 2. Worst-performing category ----------
category_sales = final_data.groupby("category")["total_order_value"].sum().sort_values()

worst_category = category_sales.idxmin()
best_category = category_sales.idxmax()

print("Category Sales:\n", category_sales, "\n")
print("Worst Performing Category:", worst_category)
print("Best Performing Category:", best_category, "\n")


# ---------- 3. High-risk customers ----------
freq = final_data.groupby("customer_id")["order_id"].nunique()
clv = final_data.groupby("customer_id")["total_order_value"].sum()

risk_df = pd.DataFrame({
    "recency": recency,
    "frequency": freq,
    "clv": clv
})

high_risk = risk_df[
    (risk_df["recency"] > 30) &
    (risk_df["frequency"] <= 2) &
    (risk_df["clv"] < risk_df["clv"].median())
]

print("Total High-Risk Customers:", len(high_risk), "\n")


# ---------- 4. Growth opportunities ----------
top_customers = clv.sort_values(ascending=False).head(10)

print("Top 10 Customers (High Value):\n", top_customers, "\n")


# ---------- 5. Strategic insights ----------
print("Key Insights:")
print(" Focus on", top_segment, "segment for maximum revenue growth")
print(" Improve performance of", worst_category, "category")
print(" Retain high-risk customers through targeted offers")
print(" Expand successful category:", best_category)
print(" Build loyalty programs for top customers")

print("\nBusiness Insights Completed ")

'''Section 20: Final Dashboard and Report
•	Create a combined analytics dashboard
•	Include key performance indicators (KPIs)
•	Apply visual storytelling techniques
•	Write insights summary
•	Prepare presentation for stakeholders
'''

print("Creating Final Dashboard...\n")

# ---------- 1. Prepare dashboard data ----------
daily_sales = final_data.groupby("order_date")["total_order_value"].sum().sort_index()

category_sales = final_data.groupby("category")["total_order_value"].sum().sort_values(ascending=False)

top_customers = final_data.groupby("customer_id")["total_order_value"].sum().sort_values(ascending=False).head(10)

segment_counts = customer_segment["final_segment"].value_counts()

engagement_counts = engagement_level.value_counts()


# ---------- 2. Create combined analytics dashboard ----------
fig, axs = plt.subplots(3, 2, figsize=(16, 14))

axs[0, 0].plot(daily_sales.index, daily_sales.values)
axs[0, 0].set_title("Daily Sales Trend")
axs[0, 0].set_xlabel("Date")
axs[0, 0].set_ylabel("Revenue")
axs[0, 0].tick_params(axis="x", rotation=45)

axs[0, 1].bar(category_sales.index, category_sales.values)
axs[0, 1].set_title("Category-wise Sales")
axs[0, 1].set_xlabel("Category")
axs[0, 1].set_ylabel("Revenue")
axs[0, 1].tick_params(axis="x", rotation=45)

axs[1, 0].bar(segment_counts.index, segment_counts.values)
axs[1, 0].set_title("Customer Segment Distribution")
axs[1, 0].set_xlabel("Segment")
axs[1, 0].set_ylabel("Number of Customers")

axs[1, 1].bar(top_customers.index.astype(str), top_customers.values)
axs[1, 1].set_title("Top 10 Customers by Revenue")
axs[1, 1].set_xlabel("Customer ID")
axs[1, 1].set_ylabel("Revenue")
axs[1, 1].tick_params(axis="x", rotation=45)

axs[2, 0].hist(final_data["total_order_value"], bins=20)
axs[2, 0].set_title("Order Value Distribution")
axs[2, 0].set_xlabel("Order Value")
axs[2, 0].set_ylabel("Frequency")

axs[2, 1].bar(engagement_counts.index, engagement_counts.values)
axs[2, 1].set_title("Engagement Level Distribution")
axs[2, 1].set_xlabel("Engagement Level")
axs[2, 1].set_ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ---------- 3. Key Performance Indicators ----------
total_revenue = final_data["total_order_value"].sum()
total_customers = final_data["customer_id"].nunique()
total_orders = final_data["order_id"].nunique()
avg_order_value = final_data["total_order_value"].mean()
retention_rate = (active_users / total_customers) * 100

print("Key Performance Indicators:\n")
print("Total Revenue:", total_revenue)
print("Total Customers:", total_customers)
print("Total Orders:", total_orders)
print("Average Order Value:", avg_order_value)
print("Retention Rate:", retention_rate, "%")


# ---------- 4. Insights Summary ----------
print("\nInsights Summary:")
print("- Sales show clear trends with peak periods and seasonal patterns.")
print("- Few high-value customers contribute a significant share of total revenue.")
print("- Repeat customers are more valuable than one-time buyers.")
print("- Certain product categories perform strongly, while others need improvement.")
print("- Customer retention and engagement strategies can help improve long-term revenue.")

print("\nFinal Dashboard and Report Completed ")
