

# doing it on iris flower dataset (UCI REPOSITORY)

# Que 1: Import the required Python libraries:
import pandas as pd

# Que 2 & 3: Locate and load dataset
# Dataset link from UCI: https://archive.ics.uci.edu/ml/datasets/iris
#as dataset is not in csv, we use seperator.

irisdata = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, sep=',')
print(irisdata.head())

# Que 4: Data Preprocessing: check for missing values in the data using pandas isnull(), describe() function to get some initial statistics. Provide variable descriptions. Types of variables etc. Check the dimensions of the data frame. 

# check for missing values:  iris is clean dataset, so no missing value
misval= irisdata.isnull().sum()
print(misval)

# get initial statistics: sl, sw, pt, pw
stats=irisdata.describe()
print(stats)

# Que 5: Data Formatting and Data Normalization: Summarize the types of variables by checking the data types 

datatypes=irisdata.dtypes
print(datatypes)

df=irisdata.info()
print(df)
# final is class variable, so is object, rest all are float64

# Que 6: Turn categorical variables into quantitative variables in Python In addition to the codes and outputs

# To turn the categorical class variable into a quantitative variable, we can encode it using one-hot encoding. This will create separate columns for each class with binary values indicating the presence or absence of the class.

quant_data=irisdata[4].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2],inplace=True)
print(irisdata)

