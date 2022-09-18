import numpy as np

# Array are fast compare to other libraries.
# Purely is array based mainly used in math operations.
# Inputs and outputs the array of numbers.

arr_1 = [20, 6, 4]

# Arrays are faster than list.
arr_2 = np.array(arr_1)

print(type(arr_1), arr_1)
print(type(arr_2), arr_2)

# Prob: add value of 10 to each element of the array.
print(arr_2 + 10 * 100)
print()

# We can convert the set, tuple, dictionaries (either keys or values) to an array.


# Unlike list and tuple, array always follow one data type.
# It converts all elements to the common data type.
# Mixed data types are not allowed. All will integer, float, or string.
# The order or preference for conversion is as follows:
# String, Float, and Integer.
arr_1 = ['20', 6, 4]
arr_2 = np.array(arr_1)
print(type(arr_2), arr_2)
print()

# Float preference.
arr_1 = [20, 6, 4.5]
arr_2 = np.array(arr_1)
print(type(arr_2), arr_2)
print()

# Forcing the data types
arr_1 = ['20', 6, 4]
arr_2 = np.array(arr_1, dtype=int)
print(type(arr_2), arr_2)
print()

# Forcing the data types
arr_1 = [20, 36, 5, '4.66']
arr_2 = np.array(arr_1, dtype=float)
print(type(arr_2), arr_2)
print()

# Forcing the data types
# This does not work as 'int' does not know about decimal point.
# So, it can not round off the float to an integer.
arr_1 = [20, 36, 5, '4.66']
arr_2 = np.array(arr_1, dtype=float)
print(type(arr_2), arr_2)
print()

# Explicit/literal array values.
arr_2 = np.array((23, 45, 56, 26), dtype=float)
arr_2 += 10
print(type(arr_2), arr_2)
print()

# Explicit/literal set values.
arr_2 = np.array({1, 1, 1, 1})
print(type(arr_2), arr_2)
print()

# Typecasting of the predefined array.
arr_1 = np.array((23, 45, 56, '26'))
print(type(arr_1), arr_1)
arr_2 = arr_1.astype(int)
print(type(arr_2), arr_2)
print()

# 2-d array.
arr_1 = np.array([[1001, 13, 19, 1],
                  [2002, 23, 29, 2],
                  [3003, 33, 39, 3]])
print("Dimension of 2d array: ", arr_1.ndim)
print("Shape (Row, Col) of 2d array: ", arr_1.shape)
print("Total elems of 2d array: ", arr_1.size)
print(arr_1)
print()

# 2-d array with mismatch values.
# Rather than converting it into 2-D array, it converted to a 1-D array of list.
# where each element in the array is a list and each list is a variable length list.

# This is not a proper form to 2-D array.
arr_1 = np.array([[1001, 13, 19, 1],
                  [2002, 23, 29],
                  [3003, 33, 39, 3]])
print("Dimension of 2d array: ", arr_1.ndim)
print("Shape (Row, Col) of 2d array: ", arr_1.shape)
print("Total elems of 2d array: ", arr_1.size)
print(arr_1)
print()

# Where condition in array
arr_1 = np.array([[10, 20],
                  [30, 40],
                  [50, 60]])
print("2d array, where (elem >= 30): \n", np.where(arr_1 >= 30))
print()

arr_1 = np.array([[10, 20],
                  [30, 40],
                  [50, 60]])
print("2d array, where (elem >= 30, 0, 1): \n", np.where(arr_1 >= 30, 0, 1))
print()

arr_1 = np.array([[10, 20],
                  [30, 40],
                  [50, 60]])
print("2d array, where (elem >= 30, 0): \n", np.where(arr_1 >= 30, arr_1, 0))
print()

# Conditions can be clubbed together.
arr_1 = np.array([[10, 20],
                  [30, 40],
                  [50, 60]])
print("2d array, where (elem >= 30 & elem <= 50, 0): \n", np.where((arr_1 >= 30) & (arr_1 <=50), arr_1, 0))
print()

# A where clause can use different arrays to form a condition, however, the shape and size
# of the array must be same.
arr_1 = np.array([[10, 20],
                  [30, 40],
                  [50, 60]])
print("2d arr_1: \n", arr_1)
arr_2 = arr_1 + 100
print("2d arr_2: \n", arr_2)
print("2d array, where (arr_1 elem >= 30, arr_1, arr_2): \n", np.where((arr_1 >= 30), arr_2, arr_1))
print()
