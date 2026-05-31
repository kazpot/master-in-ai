import pandas as pd
import numpy as np

# pandas series
groceries = pd.Series(data = [30, 6, 'Yes', 'No'], index = ['eggs', 'apples', 'milk', 'bread'])
print(groceries)

print(groceries.shape)
print(groceries.ndim)
print(groceries.size)
print(groceries.index)
print(groceries.values)

print("bananas" in groceries)
print("bread" in groceries)

# accessing and deleting elements in pandas series
print(groceries["eggs"])

print(groceries[["milk", "eggs"]])

print(groceries[0])
print(groceries[-1])
print(groceries[[0, 1]])

print(groceries.loc[["eggs", "apples"]])

print(groceries.iloc[[2, 3]])

groceries["eggs"] = 2
print(groceries)

groceries.drop("apples")
print(groceries)

groceries.drop("apples", inplace=True)
print(groceries)

fruits = pd.Series([10, 6, 3], ["apples", "oranges", "bananas"])
print(fruits)

print(fruits + 2)
print(fruits - 2)
print(fruits * 2)
print(fruits / 2)

print(np.sqrt(fruits))
print(np.exp(fruits))
print(np.power(fruits, 2))

print(fruits["bananas"] + 2)
print(fruits.iloc[0] - 2)
print(fruits[["apples", "oranges"]] * 2)
print(fruits.loc[["apples", "oranges"]] / 2)

print(groceries * 2)

distance_from_sun = [149.6, 1433.5, 108.2, 227.9, 778.6]
planets = ['Earth','Saturn', 'Venus', 'Mars', 'Jupiter']
dist_planets = pd.Series(data = distance_from_sun, index = planets)
time_light = dist_planets / 18
close_planets = time_light[time_light < 40]
print(close_planets)
