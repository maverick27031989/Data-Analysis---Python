# DATA CLEANING

# learning reference
# https://towardsdatascience.com/data-cleaning-in-python-the-ultimate-guide-2020-c63b88bf0a0d

# data set reference
# https://www.kaggle.com/benroshan/factors-affecting-campus-placement

# https://github.com/BenRoshan100/Campus-Recruitment-Analysis-Classification-models/blob/master/you-re-hired-analysis-on-campus-recruitment-data.ipynb


# ---------------------------------------IMPORTING LIBRARIES BEGINS-------------------------------------------------
# import packages

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.options.mode.chained_assignment = None

# ---------------------------------------IMPORTING LIBRARIES ENDS-------------------------------------------------


# ----------------------------READING AND UNDERSTANDING DATA FRAMES BEGINS----------------------------------------

# read the data
# sample data downloaded from Kaggle - Placement_Data_Full_Class.csv
df = pd.read_csv('C:/Users/AJAY/Desktop/fifa-data-set/Python-Flask/Placement_Data_Full_Class.csv')

# Check first Six records of data
print(df.head())

# shape and data types of the data
print(df.shape)
print(df.dtypes)

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print('Numeric Columns : ', numeric_cols)

# select non numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print('Non Numeric Columns : ', non_numeric_cols)

# ----------------------------READING AND UNDERSTANDING DATA FRAMES ENDS----------------------------------------


# ----------------------------------CHECK FOR MISSING DATA BEGINS-----------------------------------------------

# CHECK FOR MISSING DATA

# Technique #1: Missing Data Heatmap
# When there is a smaller number of features, we can visualize the missing data via heatmap.

cols = df.columns[:15]  # first 30 columns
colours = ['#000099', '#ffff00']  # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

# below command to display the plot
plt.show()

# Technique #2: Missing Data Percentage List

# if it's a larger dataset and the visualization takes too long can do this.
# % of missing.
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100)))
# result display that 31% are missing values in salary columns


# Technique #3: Missing Data Histogram

# first create missing indicator for features with missing data
for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing

# then based on the indicator, plot the histogram of missing values
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)
df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')

# below command to display the plot
plt.show()

# Technique #4: Missing Data Count in Columns
print('Columns with NULL values:', df.isnull().sum(), sep='\n')

# It can be INFERRED from result set
# 1) 67 NULL values in Salary i.e candidates with no salary
# 2) We can't use mean/median values as it impact the overall result. Can be a candidate not placed but having salary
# 3) We can put Zero (0) in the empty data, zero income.


# ----------------------------------CHECK FOR MISSING DATA ENDS-----------------------------------------------


# ----------------------------------CLEANING THE DATA BEGINS--------------------------------------------------

# SOLUTION #1: Drop the Observation
# a) drop columns with huge missing values.
# b) since we have only 1 column with limited missing values, we can keep it.
# c) Since salary is important factor to determine the status so we have to keep it


# SOLUTION #2: Drop the Feature
# a) drop columns with no dependencies.
# b) since in our data set board of education doesnt matter, we can drop them.
# c) Serial No. is also not required

df.drop(['sl_no', 'ssc_b', 'hsc_b', 'salary_ismissing', 'num_missing'], axis=1, inplace=True)
print(df.head())

# SOLUTION #3: CLEANING MISSING VALUES

# putting Zero (0) in the salary column for empty data
df['salary'].fillna(value=0, inplace=True)

# printing if salary column still has NULL or Empty values
print('Salary column with null values:', df['salary'].isnull().sum(), sep='\n')

# REPEAT AGAIN :: Technique #4: Missing Data Count in Columns
print('Columns with NULL values:', df.isnull().sum(), sep='\n')

print(df.head())  # Final check to see rows and columns information

# ----------------------------------CLEANING THE DATA ENDS--------------------------------------------------


# ------------------------------IRREGULAR DATA OUTLIERS BEGINS------------------------------------------------

# Outliers are data that is distinctively different from other observations. They could be real outliers or mistakes.
# Depending on whether the feature is numeric or categorical, we can use different techniques --
#  -- to study its distribution to detect outliers.

# print(plt.style.available)            Check how many types of style are available for plot

# Technique #1: Histogram/Box Plot

# when the feature is numeric, we can use box plot to detect outliers

with plt.style.context('Solarize_Light2'):
    ax = plt.subplot(2, 2, 1)
    df.boxplot(column=['ssc_p'])
    ax.set_title('Secondary School(%)')

    ax = plt.subplot(2, 2, 2)
    df.boxplot(column=['hsc_p'])
    ax.set_title('Higher Secondary School(%)')

    ax = plt.subplot(2, 2, 3)
    df.boxplot(column=['degree_p'])
    ax.set_title('UG Degree(%)')

    ax = plt.subplot(2, 2, 4)
    df.boxplot(column=['etest_p'])
    ax.set_title('Employability(%)')
plt.show()

# checking outliers for MBA percentage
with plt.style.context('Solarize_Light2'):
    ax = plt.subplot(2, 2, 1)
    df.boxplot(column=['mba_p'])
    ax.set_title('MBA(%)')
plt.show()  # RESULT - Only higher secondary school have mostly Outliers

# Technique #2: Descriptive Statistics
print(df['hsc_p'].describe())  # values below min value and beyond max values are outliers. i.e. need to be removed

# Technique #3: Bar Chart
# When the feature is categorical. We can use a bar chart to learn about its categories and distribution.
# bar chart -  distribution of a categorical variable
df['hsc_s'].value_counts().plot.bar()
plt.show()

# sample code to remove outliers
# https://datascience.stackexchange.com/questions/54808/how-to-remove-outliers-using-box-plot

Q1 = df['hsc_p'].quantile(0.25)
Q3 = df['hsc_p'].quantile(0.75)
IQR = Q3 - Q1  # IQR is interquartile range.

filter = (df['hsc_p'] >= Q1 - 1.5 * IQR) & (df['hsc_p'] <= Q3 + 1.5 * IQR)

# new variable to store filtered data after removing outliers
filtered_data = df.loc[filter]

with plt.style.context('Solarize_Light2'):
    ax = plt.subplot(1, 2, 1)
    df.boxplot(column=['hsc_p'])
    ax.set_title('With Outliers - hsc_p(%)')

    ax = plt.subplot(1, 2, 2)
    filtered_data.boxplot(column=['hsc_p'])
    ax.set_title('Without outliers - hsc_p(%)')
plt.show()

print(filtered_data.head())  # check the data after removing outliers

# ----------------------------------IRREGULAR DATA OUTLIERS ENDS------------------------------------------------


# ----------------------------------Visualization and Plots BEGINS--------------------------------------------------

# REFERENCE : https://seaborn.pydata.org/generated/seaborn.countplot.html

with plt.style.context('Solarize_Light2'):
    ax = plt.subplot(2, 3, 1)
    sns.countplot(x='gender', data=filtered_data, hue="status", palette="Set3")
    ax.set_title('Total Count by Gender')

    ax = plt.subplot(2, 3, 2)
    sns.countplot(x='hsc_s', data=filtered_data, hue="status", palette="Set3")
    ax.set_title('Total Count by High School Subjects')

    ax = plt.subplot(2, 3, 3)
    sns.countplot(x='degree_t', data=filtered_data, hue="status", palette="Set3")
    ax.set_title('Total Count by Degree Type')

    ax = plt.subplot(2, 3, 4)
    sns.countplot(x='workex', data=filtered_data, hue="status", palette="Set3")
    #    ax.set_title('Total Count by Work Experience')

    ax = plt.subplot(2, 3, 5)
    sns.countplot(x='specialisation', data=filtered_data, hue="status", palette="Set3")
    #   ax.set_title('Total Count by specialisation')

    ax = plt.subplot(2, 3, 6)
    sns.countplot(x='status', data=filtered_data, hue="status", palette="Set3")
#   ax.set_title('Status(Placed/Not Placed)')
plt.show()



