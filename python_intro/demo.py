import useful_functions as uf
# from module_name import object_name
# from module_name import first_object, second_object
# import module_name as new_name
# from module_name import object_name as new_name
# import package_name.submodule_name

# Python Standard Libary
# Third-Party Libraries


scores = [88, 92, 79, 93, 85]

mean = uf.mean(scores)
curved = uf.add_five(scores)

mean_c = uf.mean(curved)

print("Scores:", scores)
print("Original Mean:", mean, "New Mean:", mean_c)

print(__name__)
print(uf.__name__)