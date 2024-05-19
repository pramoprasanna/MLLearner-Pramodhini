import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('StudentsPerformance.csv')
print(df)
# Check Missing Values
print(df.isna().sum())
#Check Duplicates
print(df.duplicated().sum())
#Check Datasets
print(df.info)
#Check the number of unique values in each column
print(df.nunique())
#Stats
print(df.describe())


# EXPLORING DATA

print("Categories in gender variable:", end=" ")
print(df['gender'].unique())
print("Categories in race/ethnicity variable:", end=" ")
print(df['race/ethnicity'].unique())
print("Categories in parental level of eduction variable:", end=" ")
print(df['parental level of education'].unique())
print("Categories in lunch variable:", end=" ")
print(df['lunch'].unique())
print("Categories in test preparation course variable:", end=" ")
print(df['test preparation course'].unique())


#Define Numerical and Categorical Columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O'] #CAPITAL ALPHABET O
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

#print Columns
print('We have {} numerical features : {}'.format(len(numeric_features),numeric_features))
print('We have {} categorical features : {}'.format(len(categorical_features),categorical_features))


# ADD COLS FOR TOTAL SCORE AND AVERAGE

df['total_score'] = df['math score'] + df['reading score']+df['writing score']
df['average'] = df['total_score'] / 3
print(df.head())

reading_full = df[df['reading score'] == 100]['average'].count()
writing_full = df[df['writing score'] == 100]['average'].count()
math_full = df[df['math score'] == 100]['average'].count()

print(f'Number of students with full marks in Maths:{math_full}')
print(f'Number of students with full marks in Writing:{writing_full}')
print(f'Number of students with full marks in Reading:{reading_full}')


reading_less_20 = df[df['reading score'] <= 20]['average'].count()
writing_less_20 = df[df['writing score'] <= 20]['average'].count()
math_less_20 = df[df['math score'] <= 20]['average'].count()

print(f'Number of students with less than 20 marks in Maths:{math_less_20}')
print(f'Number of students with less than 20 marks in Writing:{writing_less_20}')
print(f'Number of students with less than 20 marks in Reading:{reading_less_20}')

