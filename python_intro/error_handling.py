try:
    print("try")
    raise ZeroDivisionError
except ZeroDivisionError as e:
    print("ZeroDivisionError occurred: {}".format(e))