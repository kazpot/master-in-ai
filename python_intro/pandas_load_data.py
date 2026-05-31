import pandas as pd

google_stock = pd.read_csv("./GOOG.csv")
print(type(google_stock))
print(google_stock.shape)

# first 5 rows
print(google_stock.head())

# last 5 rows
print(google_stock.tail())

print(google_stock.tail(8))

print(google_stock.head(2))

print(google_stock.isnull().any())

print(google_stock.describe())

print(google_stock["Adj Close"].describe())

print(google_stock.max())
print(google_stock.mean(numeric_only=True))
print(google_stock.min())
print(google_stock["Close"].min())

print(google_stock.corr(numeric_only=True))

data = pd.read_csv("./fake_company.csv")
print(data)

print(data.groupby(["Year"])["Salary"].sum())
print(data.groupby(["Year"])["Salary"].mean())
print(data.groupby(["Name"])["Salary"].sum())

print(data.groupby(["Year", "Department"])["Salary"].sum())