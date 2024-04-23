import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

np.seterr(divide = 'ignore') 

#so we can view all the rows/columns and is not restricted to 10
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Import listings.csv data
#obtained from http://insideairbnb.com/get-the-data/
#San Francisco, California, United States
#as of 04 December, 2022

#read the csv file
airbnb_data = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')


#view the data
airbnb_data.head(5)

#get a basic understanding of the dataset

#number of rows in the data set
print("\nThe number of rows in the dataset: " + str(len(airbnb_data)))
#WE HAVE 6789 rows

#get number of columns
column_names = list(airbnb_data.columns)

print("\nThe number of columns in the dataset: " + str(len(column_names)))
#WE HAVE 75 COLUMNS

#get list of columns
print("\nThe list of columns in the dataset are:")
for column_names in column_names: 
    print(column_names)

#get a basic understanding of the dataset

#get a list of just the numeric columns in the dataset
numeric_columns = airbnb_data._get_numeric_data().columns.values

print("\nThe number of numeric columns in the dataset: " + str(len(numeric_columns)))
#WE HAVE 40 NUMERIC COLUMNS

print("\nThe list numeric columns in the dataset are:")
for numeric_columns in numeric_columns: 
    print(numeric_columns)

#subset data having numeric columns

data = airbnb_data.select_dtypes(include=np.number)
data.head(5)

#since we can see a lot of NaN values above

#identify number of columns with NaN values and the number of NaN values
print(data.isnull().sum())