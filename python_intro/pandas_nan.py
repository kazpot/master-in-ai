import pandas as pd

# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8, 'suits':45},
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5, 'shirts': 2, 'shoes':5, 'suits':7},
          {'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes':10}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])

# We display the DataFrame
print(store_items)

x = store_items.isnull().sum().sum()
print(x)

print(store_items.count())

print(store_items.dropna(axis=0))
print(store_items.dropna(axis=1))

print(store_items.fillna(0))

print(store_items.ffill(axis=0))
print(store_items.ffill(axis=1))

print(store_items.bfill(axis=0))

print(store_items.interpolate(method="linear", axis=0))
print(store_items.interpolate(method="linear", axis=1))