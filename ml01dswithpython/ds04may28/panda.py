import pandas as pd
import numpy as np

# Important library to learn:
#   data science,
#   data preprocessing,
#   making machine learning models
#   data cleaning

# To perform exploratory data analysis (EDA), we use Pandas.
# Pandas library help to bring the data into python application for processing.
# Read data from xls, csv, database, etc.

# Pandas library can work on individual column (a.k.a. series) of the table.
# Entire table values when read into memory, is called as Data Frame.
# Pandas do not understand arrays unless data is read into a Dataframe (2D) or Series (1D)

# For very large data set, pyspark is used instead of pandas.

ar1 = np.array([10, 20, 30, 40, 60])
print("Array: \n", ar1)
sr1 = pd.Series(ar1)
print("Series: \n", sr1)
print()

# Pandas series can accept list, tuple as inputs.
# Pandas follow the same data types conversion as in an array.
# The data type of 'string' is called as 'object'
sr1 = pd.Series((10, 'PY', 30, 40, 60))
print("Series: \n", sr1)
print()

# Even with one float, the whole Series is converted to float.
sr1 = pd.Series((10, 20.4, 30, 40, 60))
print("Series: \n", sr1)
print()

# Even with one string form of int, the whole Series can be converted to int.
sr1 = pd.Series((10, '20', 30, 40, 60)).astype(dtype=int)
print("Series with dtype=int: \n", sr1)
print()

# Behavior of Series changes for the dictionary data type.
# The keys comes to be indexes and the values becomes the column.
d1 = {'a': 40, 'b': 20, 'c': 30, 'd': 40}
d2 = pd.Series(d1)
print("Series with dictionary: \n", d2)
print()

# Custom indexes in series. However, indexes are very rarely given.
sr1 = pd.Series((12, 13, 14), index=[1, 2, 3])
print("Series with custom index: \n", sr1)
print()

sr1 = pd.Series(6.32, index=range(5))
print("Series with single value and custom index: \n", sr1)
print()

# Picking up or indexing specific value from a series.
sr1 = pd.Series((11, 12, 13), index=[1, 2, 3])
print("Indexing a value from a series: \n", sr1[1])
print("Indexing a value from a series: \n", sr1[1:3])
print()

# Operations on the series.
# Adding series.
# Series with different length, added series contains NaN (not a number) values.
# The entire data type is changed to 'float' if NaN occurs in the series.
sr1 = pd.Series((11, 12, 13, 14))
sr2 = pd.Series((1, 2, 3))
sr3 = sr1 + sr2
print("Addition of two series: \n", sr3)
print()


# Data frames in pandas
df1 = pd.DataFrame(np.array([[1, 2, 3],
                            ['50', 60, 70]]))
print("Data frame:\n", df1)
print("Data frame data types (column wise): \n", df1.dtypes)
print()

# Mixed data types in data frame
df1 = pd.DataFrame([[10, 20, 30],
                    ['abc', 50, 60]])
print("Data frame data types (column wise): \n", df1.dtypes)
print()

# Display detailed info of the dataframe
df1 = pd.DataFrame([[10, 20, 30, 40],
                    ['abc', 50, 60]])
print("Data frame: \n", df1)
print("Data frame info: \n", df1.info())
print()

# Display brief info about data frame.
df1 = pd.DataFrame([[np.NaN, 20, 30],
                    [np.NaN, 50, 60]], columns=['A', 'B', 'C'])
print("Data frame: \n", df1)
print("Data frame info with count of NaN values: \n", df1.isna().sum())
print()


# Transpose of data frame
df1 = pd.DataFrame(np.arange(20, 60).reshape(5, 8), columns=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
print("Data frame: \n", df1)
print()
print("Data frame after transpose: \n", df1.transpose())
print()


# Building a data frame from multiple series
sr1 = pd.Series([10, 20, 30], index=['python', 'c++', 'c#'])  # list index
sr2 = pd.Series(np.array(['1', '2', 'ab']), index=['python', 'c++', 'c#'])  # array index
sr3 = pd.Series({'python': 100, 'c++': 200, 'c#': 300})  # dictionary index

df1 = pd.DataFrame({'A': sr1, 'B': sr2, 'C': sr3})
print("Data frame from series: \n", df1)
print()
