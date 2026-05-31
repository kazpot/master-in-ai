import pandas as pd

items = {'Alice': pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants']),
         'Bob': pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch'])}

print(type(items))

shopping_carts = pd.DataFrame(items)
print(shopping_carts)

data = {'Alice': pd.Series(data = [40, 110, 500, 45]),
         'Bob': pd.Series(data = [245, 25, 55])}

df = pd.DataFrame(data)
print(df)

print(shopping_carts.index)
print(shopping_carts.columns)
print(shopping_carts.values)
print(shopping_carts.shape)
print(shopping_carts.ndim)
print(shopping_carts.size)

bob_shopping_cart = pd.DataFrame(items, columns=["Bob"])
print(bob_shopping_cart)

sel_shopping_cart = pd.DataFrame(items, index=["pants", "bike"])
print(sel_shopping_cart)

alice_sel_shopping_cart = pd.DataFrame(items, index=["pants", "bike"], columns=["Alice"])
print(alice_sel_shopping_cart)

data = {"Floats": [4.5, 8.2, 9.6], "Integers": [1, 2, 3]}
df = pd.DataFrame(data, index=["label 1", "label 2", "label 3"])
print(df)

data = [{"bikes": 20, "pants": 20, "watches": 35}, {"watches": 10, "glasses": 15, "pants": 5}]
store_items = pd.DataFrame(data, index=["store1", "store2"])
print(store_items)

# accessing columns
print(store_items[["bikes"]])
print(store_items[["bikes", "pants"]])

# accessing rows
print(store_items.loc[["store1"]])

# specific cell (column label -> row label)
print(store_items["bikes"]["store1"])

store_items["shirts"] = [15, 2]
print(store_items)

store_items["suits"] = store_items["shirts"] + store_items["pants"]
print(store_items)

new_items = [{"bikes": 20, "pants": 30, "watches": 35, "glasses": 4}]
new_store = pd.DataFrame(new_items, index=["store3"])
print(new_store)

store_items = pd.concat([store_items, new_store])
print(store_items)

store_items["new_watches"] = store_items["watches"][1:]
print(store_items)

store_items.insert(5, "shoes", [8, 5, 0])
print(store_items)

store_items.pop("new_watches")
print(store_items)

store_items = store_items.drop(["watches", "shoes"], axis=1)
print(store_items)

store_items = store_items.drop(["store1", "store2"], axis=0)
print(store_items)

store_items = store_items.rename(columns={"bikes":"hats"})
print(store_items)

store_items = store_items.rename(index={"store3":"last store"})
print(store_items)

store_items = store_items.set_index("pants")
print(store_items)