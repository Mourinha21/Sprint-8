#Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
print(os.getcwd())

#Loading the dataframes

visits_logs = pd.read_csv(BASE_DIR / 'data' / 'visits_log_us.csv', parse_dates=['Start Ts', 'End Ts'])
orders_logs = pd.read_csv(BASE_DIR / 'data' / 'orders_log_us.csv', parse_dates=['Buy Ts'])
costs_logs = pd.read_csv(BASE_DIR / 'data' / 'costs_us.csv', parse_dates=['dt'])

#Converting column str -> category

visits_logs['Device'] = visits_logs['Device'].astype('category')

#Adding columns for each expression of time

visits_logs['session_day'] = visits_logs['Start Ts'].dt.date
visits_logs['session_week'] = visits_logs['Start Ts'].dt.isocalendar().week 
visits_logs['session_month'] = visits_logs['Start Ts'].dt.month
visits_logs['session_year'] = visits_logs['Start Ts'].dt.year

#Calculating average of active users on different expressions of time

dau_total = visits_logs.groupby('session_day').agg({'Uid': 'nunique'}).mean()
wau_total = visits_logs.groupby(['session_year','session_week']).agg({'Uid': 'nunique'}).mean()
mau_total = visits_logs.groupby(['session_year','session_month']).agg({'Uid': 'nunique'}).mean()

#Number of daily sessions

sessions_users_day = visits_logs.groupby('session_day').agg({'Uid': ['count', 'nunique']})

sessions_users_day.columns = ['n_sessions', 'n_users']
sessions_users_day['sess_per_user'] = sessions_users_day['n_sessions'] / sessions_users_day['n_users']

#Calculating average time of daily sessions

visits_logs['session_duration_min'] = (visits_logs['End Ts'] - visits_logs['Start Ts']).dt.total_seconds() / 60
asl_daily = visits_logs.groupby('session_day')['session_duration_min'].mean()

print('Users log in about: ', sessions_users_day['sess_per_user'].mean().round(2), ' times a day.')

#Grouping the first visit for each user ID

first_visit = visits_logs.groupby('Uid')['Start Ts'].min()
first_visit.name = 'first_visit'

#Adding first_visit column into the main dataframe

visits_logs = visits_logs.join(first_visit, on='Uid')

#Calculating each user's lifetime

visits_logs['lifetime'] = visits_logs['Start Ts'] - visits_logs['first_visit']

#Adding columns for month metrics by converting date columns

visits_logs['first_month'] = visits_logs['first_visit'].dt.to_period('M')
visits_logs['session_month_period'] = visits_logs['Start Ts'].dt.to_period('M')

#Calculating lifetime in months instead of days/seconds by applying .n method from dt.to_period

visits_logs['lifetime_months'] = (visits_logs['session_month_period'] - visits_logs['first_month']).apply(lambda x: x.n)

#Creating cohort from lifetime metrics

cohort = visits_logs.groupby(['first_month', 'lifetime_months']).agg({'Uid': 'nunique'}).reset_index()

#Adding entries that are from the first month of lifetime

initial = cohort[cohort['lifetime_months'] == 0][['first_month', 'Uid']]
initial = initial.rename(columns={'Uid': 'users'})

#Merging the initial data into the cohort

cohort = cohort.merge(initial, on='first_month')

#Calculating users retention

cohort['retention'] = ((cohort['Uid'] / cohort['users']) * 100).round(2)

#Creatign cohort table

retention_pivot = cohort.pivot_table(
    index='first_month',
    columns='lifetime_months',
    values='retention',
    aggfunc='sum'
)

#Plotting retention cohort

sns.set_theme(style='white')
plt.figure(figsize=(12,6))
plt.title('Retention Cohort Analysis (%)')
sns.heatmap(retention_pivot, annot=True, linewidths=1, linecolor='#EEEEEE', cmap='Greens', fmt='.1f')
plt.xlabel('Cohort Lifetime (Months)')
plt.ylabel('Cohort First Access Month')
plt.tight_layout(pad=0.5)
plt.show()

#Merging order_logs into visits_logs

ord_vis_logs = visits_logs.merge(orders_logs, left_on='Uid', right_on='Uid')

#Grouping dataframe by users' ID and measuring their starting time

time_till_order = ord_vis_logs.groupby('Uid').agg({'Start Ts': 'min', 'Buy Ts': 'min'})

#Grouping the amount of purchases per month

orders_months = ord_vis_logs.groupby('session_month').agg({'Buy Ts': 'count'})
mean_orders = orders_months['Buy Ts'].mean().round(0)

print(f'Users buy on average {mean_orders} times a month' )

#Measuring the mean revenue for each time stamp

order_vol = ord_vis_logs.groupby('Buy Ts').agg({'Revenue': 'mean'}).reset_index().round(3)

#Calculating both mean and median from revenue obtained

mean_ov = order_vol['Revenue'].mean().round(2)
median_ov = order_vol['Revenue'].median()
print(f"Average revenue for each order: {mean_ov}.\nMeadian revenue for each order: {median_ov}.")

#Creating a dataframe with the Life Time Value of each user by grouping the sum of their revenue

ltv = ord_vis_logs.groupby('Uid').agg({'Revenue': 'sum'}).reset_index()

#Calculating both mean and median from the LTV obtained

mean_ltv = ltv['Revenue'].mean().round(2)
median_ltv = ltv['Revenue'].median().round(2)
print(f"Average LTV for each order: {mean_ltv}.\nMeadian LTV for each order: {median_ltv}.")

#Adding a column with the month of each cost spent so we can use it to group

costs_logs['month'] = costs_logs['dt'].dt.month

#Grouping the sum of the costs by type and month

costs = costs_logs.groupby(['source_id', 'month']).agg({'costs': 'sum'}).reset_index()

#Calculating the Return of Investment (ROI)

total_costs = costs['costs'].sum()
total_revenue = ltv['Revenue'].sum()
roi = abs(((total_costs - total_revenue.round(2)) / total_costs).round(2))
print(f'There`s still {roi}% left to recover')

#Grouping the amount of purchases by month and year

n_orders = ord_vis_logs.groupby(['session_month', 'session_year']).agg({'Buy Ts': 'count'}).sort_values(['session_year', 'session_month']).reset_index()  

#Selecting purchases from 2017

n_orders_2017 = n_orders.query('session_year == 2017')

#Selecting purchases from 2018

n_orders_2018 = n_orders.query('session_year == 2018')

#Calculating sales average in a month

mean_sales = (n_orders['Buy Ts'].mean().round(0))
print(f'Users make about {mean_sales} purchases a month.')

#Plotting a graph to visualize the quantity of sales per month for both years

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First Graph (2017)
ax1.plot(n_orders_2017['session_month'], n_orders_2017['Buy Ts'], color='#2FA084')
ax1.axhline(mean_sales, color='#1F6F5F', linestyle='--', label=f'Mean Sales: {mean_sales}')
ax1.set(xlabel='Month', ylabel='Quantity of Purchases', title='Number of Sales (2017)')
ax1.grid(color='#EEEEEE', linestyle='--', linewidth=1)
ax1.set_ylim(0, n_orders_2017['Buy Ts'].max() * 1.1)

# Second Graph (2018)
ax2.plot(n_orders_2018['session_month'], n_orders_2018['Buy Ts'], color='#2FA084')
ax2.axhline(mean_sales, color='#1F6F5F', linestyle='--', label=f'Mean Sales: {mean_sales}')
ax2.set(xlabel='Month', ylabel='Quantity of Purchases', title='Number of Sales (2018)')
ax2.grid(color='#EEEEEE', linestyle='--', linewidth=1)
ax2.set_ylim(0, n_orders_2017['Buy Ts'].max() * 1.1)

plt.tight_layout(pad=0.5)
plt.show()

#Grouping costs per month and measuring its average

costs_month = costs_logs.groupby('month').agg({'costs': 'sum'}).reset_index()
mean_costs = costs_month['costs'].mean().round(2)

#Plotting the cost for each month

fig, ax = plt.subplots()

ax.bar(costs_month['month'], costs_month['costs'], color='#2FA084')
plt.axhline(mean_costs, color='#1F6F5F', linestyle='--', label=f'Mean Costs: {mean_costs}')
ax.set_xlabel('Month')
ax.set_ylabel('Total Costs')
ax.set_title('Total Costs by Month')
ax.grid(color='#EEEEEE', linestyle='--', linewidth=1)

plt.show()

print('Most of the expenses were made early and late in the year, most of the holidays around theses dates, are the most important holidays, such as Christmas, Thanksgiving, Valentine`s Day and New Year`s Day are a few examples.')

#Merging the number of orders with costs per month

roc = n_orders.merge(costs_month, left_on='session_month', right_on='month')
roc = roc.drop(columns='session_month')

#Calculating the % of Return of Cost (ROC) for each month

roc['costs/revenue'] = ((roc['Buy Ts'] / roc['costs']) * 100).round(2)
mean_roc = roc['costs/revenue'].mean().round(2)

#Plotting the return of sales by monthly cost

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(roc['month'], roc['costs/revenue'], color='#2FA084')
plt.axhline(mean_roc, color='#1F6F5F', linestyle='--', label=f'Mean ROMI: {mean_roc}%')
ax.set_xlabel('Month')
ax.set_ylabel('Return in %')
ax.set_title('Return of sales by monthly cost')
ax.grid(color='#EEEEEE', linestyle='--', linewidth=1)

plt.show()

#Grouping first purchase by user ID

first_orders = orders_logs.groupby('Uid').agg({'Buy Ts': 'min'}).reset_index().rename(columns={'Buy Ts': 'first_purchase'})
first_orders['first_purchase_month'] = first_orders['first_purchase'].dt.to_period('M')
user_source = visits_logs[['Uid', 'Source Id']].drop_duplicates()

#Merging first orders with user source

first_orders = pd.merge(first_orders, user_source, on='Uid')

#Adding month column to orders_logs

orders_logs['month'] = orders_logs['Buy Ts'].dt.to_period('M')

#Grouping revenue and costs from each source

revenue_df = orders_logs.groupby(['Uid', 'month']).agg({'Revenue': 'sum'}).reset_index()
buyers = first_orders.merge(revenue_df, on='Uid')
revenue_grouped = buyers.groupby(['Source Id', 'month']).agg({'Revenue': 'sum'}).reset_index()
costs_grouped = costs_logs.groupby(['source_id', 'month']).agg({'costs': 'sum'})

#Creating a report for cost and revenue per month

report = pd.merge(costs_grouped, revenue_grouped, left_on='source_id', right_on='Source Id')

#Creating a new dataframe with a column containing the life time value

cohort['ltv'] = report['Revenue'] / report['Source Id']

#Resetting initial dataframes

visits = pd.read_csv(BASE_DIR / 'data' / 'visits_log_us.csv')
orders = pd.read_csv(BASE_DIR / 'data' / 'orders_log_us.csv')
costs = pd.read_csv(BASE_DIR / 'data' / 'costs_us.csv')

orders = orders.rename(columns={'Buy Ts': 'buy_ts', 'Revenue': 'revenue', 'Uid': 'uid'})
orders['order_month'] =pd.to_datetime(orders['buy_ts'])
orders['buy_ts'] = pd.to_datetime(orders['buy_ts'])
costs['dt'] = pd.to_datetime(costs['dt'])
orders['order_month'] = orders['buy_ts'].dt.to_period('M')
orders['first_order_month'] = orders.groupby('uid')['order_month'].transform('min')
costs['costs_month'] = costs['dt'].dt.to_period('M')

#Creating the client acquisition cost dataframe by merging the costs and orders dataframes

cac_month = pd.merge(
    costs.groupby('costs_month').agg({'costs': 'sum'}),
    orders.groupby('first_order_month').agg({'uid': 'nunique'}),
    left_index=True,
    right_index=True
).reset_index()
cac_month['cac'] = (cac_month['costs'] / cac_month['uid']).round(2)
mean_cac = cac_month['cac'].mean().round(2)

#Plotting the average customer acquisition cost by month

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(cac_month['costs_month'].astype(str), cac_month['cac'], marker='o', color='#2FA084')
ax.axhline(cac_month['cac'].mean(), color='#1F6F5F', linestyle='--', label=f'Mean CAC: {cac_month["cac"].mean()}')
ax.set_xlabel('Month')
ax.set_ylabel('CAC (c.u)')
ax.set_title('Average Customer Acquisition Cost (CAC) by Month')
ax.grid(color='#EEEEEE', linestyle='--', linewidth=1)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

#Recreating cohorts dataframe

cohorts = orders.groupby(['first_order_month','order_month']).agg(
    revenue=('revenue', 'sum'),
    n_buyers=('uid', 'nunique')).reset_index()

cohorts['cohort_lifetime'] = (cohorts['order_month'] - cohorts['first_order_month']).apply(lambda x: x.n)
cohorts = cohorts.sort_values(['first_order_month', 'cohort_lifetime'])
cost_month = costs.groupby('costs_month').agg(cost=('costs', 'sum')).reset_index()
cohorts = pd.merge(cohorts, cost_month, left_on='order_month', right_on='costs_month')
cohorts['cac'] = (cohorts['cost'] / cohorts['n_buyers']).round(1)
cohorts['ltv'] = (cohorts['revenue'] / cohorts['n_buyers']).round(1)
cohorts['romi'] = (cohorts['ltv'] / cohorts['cac']).round(1)
cohorts['ltv_cumsum'] = cohorts.groupby('first_order_month')['ltv'].cumsum().round(2)

#Creating a cohort table for Return of Marketing Investment

romi_pivot = cohorts.pivot_table(
    index='first_order_month',
    columns='cohort_lifetime',
    values='romi',
    aggfunc='mean'
)
cumsum_romi = romi_pivot.cumsum(axis=1).round(2)

#Plotting the cohort

sns.set_theme(style='white')
plt.figure(figsize=(12,6))
plt.title('Return of Marketing Investment (ROMI) Cohort Analysis')
sns.heatmap(cumsum_romi, annot=True, linewidths=1, linecolor='#EEEEEE', cmap='Greens')
plt.xlabel('Cohort Lifetime (Months)')
plt.ylabel('Cohort First Order Month')

plt.show()

#Creating cohort for the life time value

ltv_cumsum = cohorts.pivot_table(
    values='ltv_cumsum',
    index='first_order_month',
    columns='cohort_lifetime',
    fill_value=0
)

#Plotting the cohort table for the LTV

sns.set_theme(style='white')
plt.figure(figsize=(12,6))
plt.title('Lifetime Value (LTV) Cohort Analysis')
sns.heatmap(ltv_cumsum, annot=True, linewidths=1, linecolor='#EEEEEE', cmap='Greens', fmt='.0f')
plt.xlabel('Cohort Lifetime (Months)')
plt.ylabel('Cohort First Order Month')

plt.show()


