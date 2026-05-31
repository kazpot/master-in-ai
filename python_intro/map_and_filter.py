# map function: apply a given function to all items in an input list
numbers = [1, 2, 5, 8, 11]
squares_map = map(lambda x: x**2, numbers) 

# filter function:
filtered_filter = filter(lambda x: x > threshold, numbers)

# Normalizing Dataset Sizes with Lambda and Map
numbers = [
    [34, 63, 88, 71, 29],
    [90, 78, 51, 27, 45],
    [63, 37, 85, 46, 22],
    [51, 22, 34, 11, 18]
]
flat = [x for row in numbers for x in row]
mn = min(flat)
mx = max(flat)

normalized_data = list(
    map(
        lambda row: list(
            map(lambda x: (x - mn) / (mx - mn), row)
        ),
        numbers
    )
)

# Filtering Datasets by Variance with Lambda and Filter
datasets = [
    [34, 63, 88, 71, 29],
    [90, 78, 51, 27, 45],
    [63, 37, 85, 46, 22],
    [51, 22, 34, 11, 18]
]
def variance(num_list):
    mean_val = sum(num_list) / len(num_list)
    return sum((x - mean_val) ** 2 for x in num_list) / len(num_list)

threshold = 400
filtered_datasets = list(filter(lambda row: variance(row) > threshold, datasets))
