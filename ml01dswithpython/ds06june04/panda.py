import pandas as pd
import numpy as np

# Dropping the rows or columns from the data frame
#

df1 = pd.DataFrame([[23, 13, 14],
                    [45, 67, 89],
                    [45, 99, 35],
                    [23, 74, 33]],
                   columns=['Age', 'Area_Code', 'Class_Code'])
print("Data frame with NaN data: \n", df1)
print()

# Remove the column given the name
df2 = df1.drop(['Area_Code', 'Class_Code'], axis=1)  # Set the inplace=True to modify the original data frame df1.
print("Data frame after dropping the column: \n", df2)
print()

# Drop the rows
df2 = df1.drop([0, 3])  # Default axis = 0 (row)
print("Data frame after dropping the row: \n", df2)
print()

# Drop a range of the row. Typically never done, but possible.
df2 = df1.drop(df1[0:3].index)  # Default axis = 0 (row)
# df.drop(i for in range (3))    # Alternate way of specifying index.
print("Data frame after dropping a range of rows: \n", df2)
print()

# Table data fir ciry population
df1 = pd.DataFrame([['New York', 'Mar', 12714],
                    ['Michigan', 'Apr', 82938],
                    ['Atlanta', 'Jan', 8161],
                    ['Boston', 'Sept', 5885],
                    ['Birmingham', 'Mar', 10000],
                    ['Canton', 'Feb', 1714],
                    ['Marios', 'Dec', 9238],
                    ['Atlantis', 'Nov', 58161],
                    ['Alaska', 'Feb', 585],
                    ['Arizona', 'Mar', 10000],
                    ['Pacific', 'Aug', 2714],
                    ['Ontario', 'June', 189238],
                    ['Toronto', 'Jan', 81651],
                    ['Delhi', 'Oct', 58855],
                    ['London', 'Apr', 1080],
                    ['Texas', 'Sep', 1714],
                    ['Washington', 'Jul', 2038],
                    ['Florida', 'Jan', 8161],
                    ['Orlando', 'Oct', 58185],
                    ['Memphis', 'Apr', 34552]],
                   columns=['Location', 'Month', 'Sales'])
print("Data frame data for city sales: \n", df1)
print()
print("Display the top 5 rows of a data frame: \n", df1.head())
print()
print("Display the bottom 5 rows of a data frame: \n", df1.tail())
print()
print("Display info of the data frame: \n", df1.info())
print()

df1['Sales'].replace(5885, df1['Sales'].mean(), inplace=True)
print("Replace specific value from a column: \n", df1)
print()

# Return a data frame having sales values greater than 50000.
df2 = df1[df1['Sales'] > 50000]
print("Data frame containing sales values greater than 50000: \n", df2)
print()

# Return a data frame having sales values greater than 50000.
df2 = df1[(df1['Month'] == 'Mar') | (df1['Month'] == 'Oct')]
print("Data frame containing the sales in month of MArch: \n", df2)
print()

# Return a data frame having sales values greater than 50000.
df2 = df1[(df1['Month'] == 'Mar') | (df1.Month == 'June') | (df1.Sales > 50000)]
print("Data frame containing the sales in month of March|June| Sales > 50000: \n", df2)
print()

# Return a data frame having sales values greater than 50000.
df2 = df1[((df1.Sales > 50000) & (df1['Month'] == 'Mar') | (df1.Month == 'June'))]
print("Data frame containing the sales in month of March|June AND Sales > 50000: \n", df2)
print()

# Return a data frame having sales values greater than 50000.
df2 = df1[(df1['Month'] == 'Mar') | (df1.Month == 'June') | (df1.Sales > 50000)][['Location', 'Sales']]
print("Data frame containing the sales in month of March|June| Sales > 50000 with specific columns: \n", df2)
print()

# Add new column containing the category based on certain sales criteria.
df1['Sales Category'] = np.where((df1.Sales > 50000) | (df1.Month == 'June'), 'A', 'B')
print("New column of sales category based on month and sales target: \n", df1)
print()

# Resetting the index of a filtered data frame. After filtering, the indexes are jumbled.
df2 = df1[(df1['Month'] == 'Mar') | (df1.Month == 'June') | (df1.Sales > 50000)][['Location', 'Sales']]
df2.reset_index(inplace=True, drop=True)
print("Resetting indexes of a jumbled data frame after filtering: \n", df2)
print()

# Changing the case of the values of specific column
df2['Location'] = df2['Location'].str.lower()
print("Changing the case of values of specific column: \n", df2)
print()

# Filtering the column values containing specific char set.
df1['Location'] = df1['Location'].str.lower()
df2 = df1[df1['Location'].str.contains('on')]  # startswirh('an') # endswith('an')
print("Filtering the column values containing specific char set: \n", df2)
print()

# Python's "by reference" also applies to Data frames. So, copy() is recommended
# if the reflection of the change in one reference is to be avoided in second one.
df11 = pd.DataFrame({'age': ['40', 20, 30, 40],
                     'salary': [50, 60, 70, 45]})
print("Data frame before reflection: \n", df11)
df2 = df11.copy()
df2['Day'] = [1, 2, 3, 4]
print("Data frame (df1) after reflection: \n", df11)
print("Data frame (df2) after reflection: \n", df2)
print()

# Applying a function to a column of a data frame
df1 = pd.DataFrame({'Location': ['Florida', 'Detroit', 'Memphis', 'Seatle', 'Florida', 'Detroit', 'Memphis', 'Seatle',
                                 'Florida', 'Detroit', 'Florida'],
                    'Age': [50, 20, 30, 40, 45, 23, 56, 45, 34, 23, 50],
                    'Salary': [50000, 60000, 20000, 45000, 343000, 232000, 50000, 50000, 45000, 50000, 50000]})
df1['Date'] = pd.to_datetime(pd.Series(['2022-08-10'] * 20))
print("Data frame before reflection: \n", df11)
print("Data frame (df1) before applying function: \n", df1)


def foo(x):
    if x > 50000:
        return 'A'
    elif 20000 < x < 50000:
        return 'B'
    else:
        return 'C'


df1['SalaryType'] = df1['Salary'].apply(foo)
print("Data frame (df1) after applying function: \n", df1)
print()

# Applying lambda functions
df1['SalaryFactor'] = df1['Salary'].apply(lambda x: 500 if x > 50000 else 100 if (20000 < x < 50000) else 50)
print("Data frame (df1) after applying lambda function: \n", df1)
print()

# Applying lambda functions - to add a new column with last 3 char of 'Location' column.
df1['Abbreviation'] = df1['Location'].apply(lambda x: x[-3:])
print("Data frame (df1) after applying lambda function: \n", df1)
# df1['Year'] = df1['Date'].astype(str).apply(lambda x: x[:4])
# df1['Year'] = df1['Date'].apply(lambda x: str(x)[:4])
df1['Year'] = df1['Date'].apply(lambda x: x.year)
# df1['Day'] = df1['Date'].apply(lambda x: x.day_name())
print("Data frame (df1) after applying lambda function for year: \n", df1)
print()

# Find out the unique values from a specific column
print("Number of unique values in 'Month' column: \n", df1)
print("Number of unique values in 'Month' column: \n", df1['SalaryType'].nunique())
print()
print("Number of unique values in 'Sales' column: \n", df1['Salary'].nunique())
print()
print("Number of unique value count (frequency) in 'Sales' column: \n", df1['Salary'].value_counts())
print()

# Print the unique values per column in a dictionary
col = ['Age', 'Salary', 'SalaryFactor', 'SalaryType']
drop_col = ['Location', 'Date']
dic1 = {i: df1[i].unique().tolist() for i in df1.drop(drop_col, axis=1)}
print("Dictionary data is: ", dic1)
print()

# Detect the duplicated data in a data frame
print("Number of duplicates in the data frame: \n", df1.duplicated().sum())
print()
df2 = df1.drop_duplicates()
print("Data frame after dropping duplicates: \n", df2)
print()
print("Number of duplicates in the data frame: \n", df2.duplicated().sum())
print()

