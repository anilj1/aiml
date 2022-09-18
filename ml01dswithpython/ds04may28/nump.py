import numpy as np

# Reshaping the array
a1 = np.arange(2.2, 13.6, 0.78)
print("Size of ranged array is: \n", a1.size)
print("Ranged array is: \n", a1)
print()

# Convert this 1D array to 2D array
# For size of 15, (3, 5), (5, 3), (15, 1), and (15, 1) 2D arrays are possible.
a2 = a1.reshape(3, 5)
print("Size of reshaped array is: \n", a2.size)
print("Reshaped array is: \n", a2)
print()

# Auto-detection of one of the dimension.
# A value of -1 means auto-detect the dimension.
a2 = a1.reshape(5, -1)
print("Size of auto-reshaped array is: \n", a2.size)
print("Auto-reshaped array is: \n", a2)
print()

# Reshaping works on an existing 2D array as well.
a2 = a2.reshape(3, -1)
print("Size of auto-reshaped 2D array is: \n", a2.size)
print("Auto-reshaped 2D array is: \n", a2)
print()

# Flattening the 2D array
a1 = a2.flatten()
print("Size of flattened 2D array is: \n", a1.size)
print("Flattened array is: \n", a1)
print()

# Unique array elements
a1 = np.array([[1, 2, 2],
               [4, 5, 5],
               [4, 8, 9]])
a2 = np.unique(a1)
print("Unique array elem are: \n", a2)
print()
u_val, u_cnt = np.unique(a1, return_counts=True)
print("Unique array elem are: \n", u_val)
print("Unique array elem count: \n", u_cnt)
print()

# Problem: write a function to generate the dictionary output of the unique numbers and their count.
print("Dictionary type: ", dict(zip(u_val, u_cnt)))

# linspace - linearly spaced element array
# (start, stop, total_elem). The function decides the step size.
a1 = np.linspace(1, 10, 50)
print("Linspace array: \n", a1)
print()

a1, step = np.linspace(1, 10, 50, retstep=True)
print("Linspace array: {} and step size: {}\n".format(a1, step))
print()

# Convert the linspace array to int type
a1 = np.linspace(1, 10, 50).astype(int)
print("Linspace array: {}:\n".format(a1))
print()

# Random function generates a random value array
a1 = np.random.random((5, 4))
print("Random value array:\n", a1)
print()

# Random function generates with a seed returns the same random value array
np.random.seed(2000)
a1 = np.random.random(5)
print("Fixed random value array:\n", a1)
print()

# Random function generates with a seed returns the same random value array
np.random.seed(2000)
a1 = np.random.randn(5)
print("Fixed random array with +ve and -ve decimal values:\n", a1)
print()

# Random function generates with a seed returns the same random value array
np.random.seed(2000)
a1 = np.random.randint(0, 10, 20)  # 40 values in the range of 0 to 10.
print("1D array of random integer of given size:\n", a1)
print()

# Random function generator of 2D array.
np.random.seed(2000)
a1 = np.random.randint(0, 10, (3, 5))
print("2D array of random integer of given size:\n", a1)
print()

# Random array generation with normal distribution
# Data with 20 observation having mean of 8 and std deviation of 2.
a1 = np.random.normal(8, 2, 20)  # normal(mean, std, observations)
print("Array of random integer of normal distribution of given size:\n", a1)
print()

# Normally distributed data
a1 = np.random.normal(0, 1, 100)  # normal(mean, std, observations)
print("Normal distribution data mean:\n", np.mean(a1))
print("Normal distribution data std:\n", np.std(a1))
print("Normal distribution data variance:\n", np.var(a1))
print("Normal distribution data size:\n", a1.size)
print()

# Transpose of the 2D array: column and rows are swapped.
a1 = np.arange(10).reshape(5, 2)
print("Array before transpose: \n", a1)
a1 = a1.transpose()
print("Array after transpose: \n", a1)
print()

# np.zeros((row, col)) - init an array of size with 0.
a1 = np.zeros((3, 4))
print("Array zeroed: \n", a1)

# np.full((row, col), elem) - fills an array of size with given number.
a1 = np.full((3, 4), 10)
print("Array zeroed: \n", a1)

# np.eye((row, col), elem) - fill the diagonal values with elem.
a1 = np.eye(4, 4)
print("Array eyed: \n", a1)

a1 = np.eye(4, 4, -1)
print("Array eyed shifted below: \n", a1)

a1 = np.eye(4, 4, 1)
print("Array eyed shifted right: \n", a1)
print()

# Indexing the array elements 
np.random.seed(5)
a1 = np.random.randint(1, 40, (4, 5))  # 40 values in the range of 0 to 10.
print("2D array of random integer of given size:\n", a1)
print("Elem at (r=5, c=4):\n", a1[2, 1])
print()
print("Only row (r=5):\n", a1[1])
print()
print("Only column (c=3):\n", a1[:, 1])
print()
print("Two rows (2:3) with column 3:\n", a1[1:3, 3])
print()
print("Two rows (2:4) with two column (3:5):\n", a1[2:4, 3:5])
print()
print("Two rows (3:) with two column (3:):\n", a1[:2, 3:])
print()

# Custom indexes
print("Custom indexes:\n", a1[[1, 2, 3], [3, 3, 3]])
print()
a1[:2, 3:] = 100
print("Filling up using indexes:\n", a1)
print()

# Min, Max, ArgMin, ArgMax, ArgWhere
a1 = np.array([[1, 2],
               [8, 19],
               [90, 5],
               [999, 0]])

print("Array:\n", a1)
print()
print("Array min:\n", np.min(a1))
print()
print("Array max:\n", np.max(a1))
print()

# arg, which returns the index always converts the 2D array into 1D.
# Argmax
print("Array argmax:\n", a1.argmax())  # [1, 2, 8, 19, 90, 5, 999, 0]
print()  # axis = 0 -> column, axis = 1 -> row
print("Array argmax axis=0:\n", a1.argmax(axis=0))  # [1, 8, 90, 999], [2, 19, 5, 0]
print()
print("Array argmax axis=1:\n", a1.argmax(axis=1))  # [1, 2] [8, 19] [90, 5] [999, 0]
print()

# Argmin
print("Array argmin:\n", a1.argmin())  # [1, 2, 8, 19, 90, 5, 999, 0]
print()
print("Array argmin axis=0:\n", a1.argmin(axis=0))  # [1, 8, 90, 999], [2, 19, 5, 0]
print()
print("Array argmin axis=1:\n", a1.argmin(axis=1))  # [1, 2] [8, 19] [90, 5] [999, 0]
print()

# Argwhere - returns the index of all elem matching the condition.
print("Array argwhere elem > 10:\n", np.argwhere(a1 > 10))
print()

# Filtering values from array
np.random.seed(5)
a1 = np.arange(1, 21).reshape(5, 4)  # 40 values in the range of 0 to 10.
print("Array: \n", a1)
print()
print(a1[a1 >= 15])
print()

# Adding array elements
np.random.seed(5)
a1 = np.arange(1, 13).reshape(4, 3)  # 40 values in the range of 0 to 10.
print("Array: \n", a1)
print("Adding whole array: \n", np.sum(a1))
print("Adding every column of the array: \n", np.sum(a1, axis=0))
print("Adding every row of the array: \n", np.sum(a1, axis=1))
print("Adding every row of the array where values are > 10: \n", np.sum(a1, axis=1, where=a1 > 10))
print()

# Cumulative sum and product
print("Array: \n", a1)
print("Array mean / average: \n", np.mean(a1))
print("Array cumulative sum: \n", np.cumsum(a1))
print("Array cumulative product: \n", np.cumprod(a1))
print()

# Rounding off the elem values
a1 = [[10.43433, 20.34423432],
      [33.554423, 55.3432423]]
print("Array: \n", a1)
print("Array round off: \n", np.round(a1))
print("Array round off to 2 decimal points: \n", np.round(a1, 2))
print()

# Getting absolute values
a1 = [[-10.43, -20.32],
      [33.53, -55.23]]
print("Array: \n", a1)
print("Array absolute: \n", np.absolute(a1))
print("Array abs: \n", np.abs(a1))
print()

# Log values of an array
np.random.seed(5)
a1 = np.arange(1, 10).reshape(3, 3)  # 40 values in the range of 0 to 10.
print("Array: \n", a1)
print("Array natural log: \n", np.log(a1))
print("Array log base 2: \n", np.log2(a1))
print("Array log base 10: \n", np.log10(a1))
print()

# Anti-log i.e. exponential
print("log of 7: ", np.log(7))
print("Anti log of 7: ", np.exp(np.log(7)))

