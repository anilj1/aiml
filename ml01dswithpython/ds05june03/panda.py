import pandas as pd
import numpy as np

# Transpose of data frame
df1 = pd.DataFrame(np.arange(81, 84).reshape(-1, 3), columns=['C1', 'C2', 'C3'])
print("Data frame: \n", df1)
print()
print("Data frame columns: \n", df1.columns)
print()

# Renaming the columns - method #1
df1.columns = ['a1', 'a2', 'a3']
print("Data frame columns: \n", df1.columns)
df1.columns += '1'
print("Data frame: \n", df1)
print()

# Changing all columns - method #2
df1.columns = ['a1', 'a2', 'a3']
print("Data frame columns: \n", df1.columns)
df1.columns = [i + '_1' for i in df1.columns]
print("Data frame: \n", df1)
print()

# Changing specific column name
df1.columns = ['a1', 'a2', 'a3']
print("Data frame columns: \n", df1.columns)
df1.rename(columns={'a1': 'new1'}, inplace=True)  # When change is to be permanent.
print("Data frame specific column: \n", df1)
print()

# Building data frame using the dictionary.
df1 = pd.DataFrame({'age': ['40', 20, 30, 40],
                    'salary': [50, 60, 70.45, 80],
                    'number': [90, 100, 110, 120],
                    'name': ['a', 'b', 'c', 'd']})
print("Data frame using dictionary: \n", df1)
print()
print("Data frame using dictionary columns: \n", df1.columns)
print()

# Modifying column names of a data frame using dictionary.
df1.columns = ['x', 'y', 'z', 'z']
print("Data frame column updates: \n", df1.columns)
print()

# Referencing / calling the columns.
# 1. using the [] operator
print("Column type using [] oper: \n", df1['x'].dtype)
print()
# 2. using the . operator. (Not recommended approach).
print("Column type using . oper: \n", df1.x.dtype)
print()
# Picking up multiple columns
print("Multiple columns using [] oper: \n", df1[['x', 'y']])
print()

# Picking up columns and save into another data frame
df2 = df1[['x', 'y']]
print("Extracted columns saved into new data frame: \n", df2)
print()

# Operations on columns
# 1. Adding a number to all elem of a column.
df1 = pd.DataFrame({'age': ['40', 20, 30, 40],
                    'salary': [50, 60, 70.45, 80],
                    'number': [90, 100, 110, 120],
                    'name': ['a', 'b', 'c', 'd']})
print("Data frame using dictionary: \n", df1)
df1['salary'] = df1['salary'] + 10
print("Data frame using dictionary: \n", df1)
print()

# Adding a new column from an existing column
df1['salary_new'] = df1['salary'] + 10
print("Data frame using dictionary: \n", df1)
print()

# Creating new column by merging the two columns.
df1['new_data'] = df1['salary'].astype(str) + '_' + df1['name']
print("A new one by merging the two column: \n", df1)
print()
df1['rent'] = [1000, 1200, 1300, 1400]
print("A new column by explicit array values: \n", df1)
print()

# Printing only teh values from a data frame i.e. no index and column names.
ar1 = np.arange(10, 121, 10).reshape(4, -1)
df1 = pd.DataFrame(ar1, columns=['Years lived', 'City_Code', 'Country_Code'])
print("Data frame with index and column: \n", df1)
print("Data frame with only its values: \n", df1.values)
print()

# Getting a specific column in array form
print("Data frame column with its values: \n", df1['Years lived'].values)
print()

# Getting data frame columns in array form
print("Data frame columns: \n", df1.columns)
print()
print("Data frame columns from keys() func: \n", df1.keys())
print()
print("Data frame total number {}, dimension {}, shape {}".format(df1.size, df1.ndim, df1.shape))
print()

# Adding a new column merging it with an existing column
ar1 = np.arange(10, 121, 10).reshape(4, -1)
df1 = pd.DataFrame(ar1, columns=['Years lived', 'City_Code', 'Country_Code'])
df1['Continent'] = ['Asia', 'NorthA', 'Africa', 'Antarctica']
df1['Cont_Country'] = df1['Continent'] + " " + df1['Country_Code'].astype(str)
print("Data frame with index and column: \n", df1)


# Data frame replace
df1 = pd.DataFrame(np.arange(20, 0, -1).reshape(5, -1),
                   index=['a','b','c','d','e'],
                   columns=['A','B','C','D'])

print("Data frame: \n", df1)
print()

# Replace a specific vlaue in a data frame
df1.replace(5, 'Five', inplace=True)
print("Data frame one-to-one replacement: \n", df1)
print()

# Replace entire column value in a data frame.
df1['A'] = 100
print("Data frame one-to-many replacement: \n", df1)
print()

# Replace a set of values in a data frame.
df1.replace([5, 6, 7], 60, inplace=True)
print("Data frame many-to-one replacement: \n", df1)
print()

# Replace a set of values in a data frame.
df1.replace([19, 18, 17], [89, 88, 87], inplace=True)
print("Data frame many-to-many replacement: \n", df1)
print()


# Dealing with NaN values in a data frame.
df1 = pd.DataFrame([[23, 13, 14],
                    [np.NaN, 67, 89],
                    [45, np.NaN, 35],
                    [23, 74, 88]],
                   columns=['Age','Area_Code','Class_Code'])
df1['State'] = ['Delhi', 'UP', np.NaN, np.NaN]
print("Data frame with NaN data: \n", df1)

# Typically, we have to find out how much data is missing.
# If for example, there are 2 records out of 1000 are missing, we can delete the two missing records.
# If there are 100+ records missing, we need to consider how we can fill in for the missing values.
# When filling for the missing values, the data is always looked at column wise.
# The other values in the column are used to decide the replacement for the NaN.

# Find out if any of the column has a NaN values.
print("NaN values in a data frame: \n", df1.isnull().sum())
print()
print("NaN values in a data frame: \n", df1.isna().sum())
print()
print("Total NaN values in the entire data frame: \n", df1.isna().sum().sum())
print()

# Drop NaN - if any value in the row is NaN, it will drop the row.
df1 = pd.DataFrame([[23, 13, 14],
                    [np.NaN, 67, 89],
                    [45, np.NaN, 35],
                    [23, 74, 88]],
                   columns=['Age','Area_Code','Class_Code'])
df1['State'] = ['Delhi', 'UP', np.NaN, np.NaN]
print("Data frame with NaN data: \n", df1)
print()
df2 = df1.dropna()
print("Data frame after dropping NaN data: \n", df2)
print()

# Drop a row if specific column has a NaN value
print("Data frame with NaN data: \n", df1)
print()
df2 = df1.dropna(how='any', subset=['Age', 'Area_Code'])
print("Drop data frame if 'Age' col is NaN data: \n", df2)
print()

# Drop the row only if all the values of the row are NaN
df2 = df1.dropna(how='all')
print("Drop data frame row only if all the col values are NaN: \n", df2)
print()

# Drop the data frame row only if there are at least thresh=3 NaN values.
df2 = df1.dropna(axis=0, how='any', thresh=3)
print("Hold data frame row if there are thresh=3 non-NaN values: \n", df2)
print()


# Fill the NaN value with a valid value. This is always done column wise.
# The NaN data values can only be inferred using the related column data.
df1['Age'].fillna(45, inplace=True)
print("Filling up NaN values in a column with a valid data: \n", df1)
print()

# Finding median, mean of a column
print("Data frame with NaN data: \n", df1)
print()
print("Print the max value from the col: \n", df1['Age'].max())
print("Print the mean/average value from the col: \n", df1['Age'].mean())
print("Print the mode (most appearing) value from the col: \n", df1['Age'].mode())
print()

# Filling the NaN value by taking the next of the previous value relative to NaN
print("Data frame with NaN data: \n", df1)
print()
print("Data frame fill NaN data with the value from next row: \n", df1.fillna(method='bfill', inplace=True))
print("Data frame after filling the NaN data with 'bfill: \n", df1)
print()
print("Data frame fill NaN data with the value from next row: \n", df1.fillna(method='ffill', inplace=True))
print("Data frame after filling the NaN data with 'bfill: \n", df1)
print()



