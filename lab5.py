import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\lab5\ITA105_Lab_5_Supermarket.csv")

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print(df.head())


print(df.isnull().sum())
df_ffill = df["revenue"].ffill()
df_bfill = df['revenue'].bfill()
df_interpolate = df.interpolate()
df_missing_demo =  pd.DataFrame({
    'pre_fill': df['revenue'],
    'ffill': df_ffill,
    'bfill': df_bfill
})
print(df_missing_demo.head(10))

df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['day_of_week'] = df.index.dayofweek  # 0=Monday
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

monthly_revenue = df['revenue'].resample('ME').sum()

monthly_revenue.plot(figsize=(10,5), title='Doanh thu theo tháng')
plt.show()
weekly_revenue = df['revenue'].resample('W').sum()

weekly_revenue.plot(figsize=(10,5), title='Doanh thu theo tuần')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df_bfill, model='additive', period=30)

result.plot()
plt.show()
result = seasonal_decompose(df_ffill, model='additive', period=30)

result.plot()
plt.show()

# -----------------------------

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\lab5\ITA105_Lab_5_Web_traffic.csv")

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

df_hourly = df.resample('h').sum()

df_hourly = df_hourly.interpolate(method='linear')

df_hourly['hour'] = df_hourly.index.hour
df_hourly['day_of_week'] = df_hourly.index.dayofweek

hourly_pattern = df_hourly.groupby('hour')['visits'].mean()

hourly_pattern.plot(kind='line', figsize=(10,5), title='Lưu lượng theo giờ trong ngày')
plt.xlabel("Hour")
plt.ylabel("Visits")
plt.show()

daily_pattern = df_hourly.groupby('hour')['visits'].mean()

daily_pattern.plot(title="Daily Seasonality (Theo giờ)")
plt.show()
weekly_pattern = df_hourly.groupby('day_of_week')['visits'].mean()

weekly_pattern.plot(kind='bar', title='Weekly Seasonality')
plt.xlabel("Day of Week (0=Mon)")
plt.show()
# ----------------------------------

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\lab5\ITA105_Lab_5_Stock.csv")

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df = df.sort_index() 

df = df.asfreq('D')

df['close_price'] = df['close_price'].ffill()

df['close_price'].plot(figsize=(12,5), title='Giá đóng cửa')
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

df['MA7'] = df['close_price'].rolling(7).mean()
df['MA30'] = df['close_price'].rolling(30).mean()

df[['close_price','MA7','MA30']].plot(figsize=(12,6))
plt.title("Trend với Moving Average")
plt.show()

monthly_pattern = df.groupby(df.index.month)['close_price'].mean()

monthly_pattern.plot(kind='bar', title='Seasonality theo tháng')
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.show()
# ------------------------------

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\lab5\ITA105_Lab_5_Production.csv")

df['week_start'] = pd.to_datetime(df['week_start'])
df.set_index('week_start', inplace=True)

df = df.sort_index()

print(df.isnull().sum())

df = df.interpolate()

df = df.ffill().bfill()

df['week'] = df.index.isocalendar().week
df['quarter'] = df.index.quarter
df['year'] = df.index.year

df['rolling_7'] = df['production'].rolling(7).mean()
df['rolling_30'] = df['production'].rolling(30).mean()

df[['production','rolling_7','rolling_30']].plot(figsize=(12,6))
plt.title("Trend sản xuất (Rolling Mean)")
plt.show()

quarter_pattern = df.groupby('quarter')['production'].mean()

quarter_pattern.plot(kind='bar', title='Seasonality theo quý')
plt.xlabel("Quarter")
plt.ylabel("Production")
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['production'], model='additive', period=12)

result.plot()
plt.show()