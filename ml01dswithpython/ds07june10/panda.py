import pandas as pd
import numpy as np

# LOC & ILOC
print("Data frame for demo of LOCs: \n", df1)
print()
df2 = df1.loc[0]
print("Location 0: \n", df2)
print()
df2 = df1.loc[0:3]
print("Location 0 to 3: \n", df2)
print()
df2 = df1.loc[[0, 3, 6]]
print("Location 0, 3, and 6: \n", df2)
print()
df2 = df1.loc[0:3, ['Salary', 'SalaryFactor']]
print("Location 0:3: \n", df2)
print()
df2 = df1.loc[[0, 3, 6], ['Salary', 'SalaryFactor']]
print("Location 0, 3, and 6: \n", df2)
print()

# ILOC
# The ILOC API only accepts the index unlike the name(s) of the column in LOC.

# Set a column as index column
df2 = df1.set_index('Location')
print("Data frane with location column as index: \n", df2.head())
print()
print("Value from data frame from 'Atlanta' and 'SalaryFactor': \n", df2.loc['Detroit', 'SalaryFactor'])
print()

# Group by - to find out the total or average 'Salary' grouped by the 'Location'.
df2 = df1.groupby('Location')['Salary'].mean()
print("Data frane data grouped by Location: \n", df2.head())
print()
df2 = df1.groupby('Location')[['Salary', 'Age']].mean()
print("Data frane data grouped by Location: \n", df2)
print()

# Group the values by the 'SalaryType' category.
df2 = df1.groupby('SalaryType')['Location'].unique()
print("Data frame data grouped by SalaryType: \n", df2)
print()
df2 = df1.groupby('SalaryType')['Location'].agg(['unique', 'nunique'])
print("Data frame data grouped by aggregate of unique and nunique of Location: \n", df2)
print()
df2 = df1.groupby('Location')['Salary'].agg(['mean', 'min', 'max'])
print("Data frame data grouped by aggregate of mean, min, and max of Salary: \n", df2)
print()
df2 = df1.groupby('Location')['Salary'].describe()
print("Data frame data grouped by aggregate of description: \n", df2)
print()
df2 = df1.groupby('SalaryType')['Year'].value_counts()
print("Data frame data grouped the value count: \n", df2)
print()

# Merging the data frames.
mathdf = pd.DataFrame({'Student': ['Tom', 'Tom', 'Jack', 'Dan', 'Ram', 'Jeff', 'David'],
                       'Id': [10, 11, 56, 31, 85, 9, 22]})
print("Math data frame: \n", mathdf)
print()
scidf = pd.DataFrame({'Student': ['Tom', 'Ram', 'David', 'Abhishek'],
                      'Tenure': [100, 850, 202, 100]})
print("Math data frame: \n", scidf)
print()

# LEFT merge
leftmdf = pd.merge(left=mathdf,
                   right=scidf,
                   on='Student',
                   how='left')
print("Left merge on student: \n", leftmdf)
print()

# RIGHT merge
rightmdf = pd.merge(left=mathdf,
                    right=scidf,
                    on='Student',
                    how='right')
print("Right merge on student: \n", rightmdf)
print()

# INNER merge
innermdf = pd.merge(left=mathdf,
                    right=scidf,
                    on='Student',
                    how='inner')
print("Inner merge on student: \n", innermdf)
print()

# OUTER merge
outmdf = pd.merge(left=mathdf,
                  right=scidf,
                  on='Student',
                  how='outer')
print("Outer merge on student: \n", outmdf)
print()

# Merge with non-matching column names.
mathdf = pd.DataFrame({'Learner': ['Tom', 'Jack', 'Dan', 'Ram', 'Jeff', 'David'],
                       'Number': [100, 56, 31, 85, 9, 22]})
print("Math data frame: \n", mathdf)
print()
scidf = pd.DataFrame({'Student': ['Tom', 'David', 'Python'],
                      'Id': [100, 22, 999999]})
print("Math data frame: \n", scidf)
print()

# Outer merge
rightmdf = pd.merge(left=mathdf,
                    right=scidf,
                    left_on='Number',
                    right_on='Id',
                    how='right')
print("Right merge with left-on, right-on Number and Id: \n", rightmdf)
print()

# After filling the NaN values, the same right merged data frame looks as below.
rightmdf['Number'].fillna(rightmdf['Number'].mean(), inplace=True)
rightmdf['Learner'].fillna('Other', inplace=True)
print("Right merge dta frame after filling NaN values: \n", rightmdf)
print()

# Merge multiple columns
rightmdf = pd.merge(left=mathdf,
                    right=scidf,
                    left_on=['Learner', 'Number'],
                    right_on=['Student', 'Id'],
                    how='left')
print("Right merge with multiple columns left-on, right-on Number and Id: \n", rightmdf)
print()

# Concatenation of data frames - row wise. (ID column exists in both the data frames)
mathdf = pd.DataFrame({'Student': ['Tom', 'Jack', 'Dan', 'Ram', 'Jeff', 'David'],
                       'Id': [10, 56, 31, 85, 9, 22]})
print("Math data frame: \n", mathdf)
print()
scidf = pd.DataFrame({'Student': ['Harry', 'Tom', 'Elon', 'Warren'],
                      'Id': [10, 85, 22, 999999],
                      'City': ['Fremont', 'Dublin', 'LosGatos', 'Milpitas']})
print("Math data frame: \n", scidf)
print()

catdf = pd.concat([mathdf, scidf], axis=0)
print("Concatenated data frame row wise: \n", catdf)
print()

# Concatenate with distinct columns (RollNum and Id columns are distinct)
mathdf = pd.DataFrame({'Student': ['Tom', 'Jack', 'Dan', 'Ram', 'Jeff', 'David'],
                       'RollNum': [10, 56, 31, 85, 9, 22]})
print("Math data frame: \n", mathdf)
print()
scidf = pd.DataFrame({'Student': ['Harry', 'Tom', 'Elon', 'Warren'],
                      'Id': [10, 85, 22, 999999],
                      'City': ['Fremont', 'Dublin', 'LosGatos', 'Milpitas']})
print("Math data frame: \n", scidf)
print()

# By default, the join is 'outer' (union).
# In the 'inner' (intersect) join, only the column(s)
# present in both the data frame are selected.
catdf = pd.concat([mathdf, scidf], axis=0, join='outer')
print("Concatenated data frame: \n", catdf)
print()

# Concatenation of data frame - column wise
# In the column wise concat, the focus is on alignment of indexes of the entries.
# The same indexes are places parallel/ next to each other. The missing columns
# are populated with NaN values.
mathdf = pd.DataFrame({'Student': ['Tom', 'Jack', 'Dan', 'Ram', 'Jeff', 'David'],
                       'RollNum': [10, 56, 31, 85, 9, 22]})
print("Math data frame: \n", mathdf)
print()
scidf = pd.DataFrame({'Student': ['Harry', 'Tom', 'Elon', 'Warren'],
                      'Id': [10, 85, 22, 999999],
                      'City': ['Fremont', 'Dublin', 'LosGatos', 'Milpitas']})
print("Math data frame: \n", scidf)
print()

catdf = pd.concat([mathdf, scidf], axis=1)
print("Concatenated data frame in 'outer' form: \n", catdf)
print()

# There is only 'inner' and 'outer' concat of the data frames.
# There is NO 'left' or 'right' concatenation.

# Inner concat only presents the common indexes present in both the data frames.
mathdf = pd.DataFrame({'Student': ['Tom', 'Jack', 'Dan', 'Ram', 'Jeff', 'David'],
                       'RollNum': [10, 56, 31, 85, 9, 22]})
print("Math data frame: \n", mathdf)
print()
scidf = pd.DataFrame({'Student': ['Harry', 'Tom', 'Elon', 'Warren'],
                      'Id': [10, 85, 22, 999999],
                      'City': ['Fremont', 'Dublin', 'LosGatos', 'Milpitas']})
print("Math data frame: \n", scidf)
print()

catdf = pd.concat([mathdf, scidf], axis=1, join='inner')
print("Concatenated data frame in 'inner' form: \n", catdf)
print()

# Saving the data frame into a CSV file on to local host.
catdf.to_csv('student_data.csv', index=False)  # index = False does not add the index column.

# Reading the CSV file from disk into a Data frame.
csvdf = pd.read_csv('student_data.csv')
print("Data frame after reading from a CSV file, before drop: \n", csvdf)
print()

# Project discussion starts at: 3:53:10.
# Use this to complete the second project related to comcast data.
import matplotlib.pyplot as plt

plt.bar(csvdf['RollNum'], csvdf['Student'])
plt.show()

csvdf['RollNum'].plot(kind='line', color='green')
plt.show()

plt.plot(np.arange(2, 50))
plt.show()
