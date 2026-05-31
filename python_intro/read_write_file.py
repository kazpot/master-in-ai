f = open('/my_path/my_file.txt', 'r')
file_data = f.read()
f.close()
print(file_data)

f = open('/my_path/another_file.txt', 'w')
f.write('Hello world')
f.close()

# special syntax taht auto-closes
with open('another_file.txt', 'r') as f:
    file_data = f.read()