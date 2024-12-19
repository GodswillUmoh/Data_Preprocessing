# Data_Preprocessing
> Data preprocessing is the process of transforming raw data into a clean, organized, and structured format suitable for analysis or modeling. It is a critical step in the data pipeline to ensure that data is accurate, consistent, and usable for deriving insights or building machine learning models.
>
> Steps in Data Preprocessing: Data Cleaning, Data Transformation, Data Reduction, Data Integration, Data Encoding

## Dataset
In this data preprocessing, I created a simple data. _Note: Data is created for learning purposes and not necessarily a fact_
|Country | Age |Salary |	Purchased|
|-------|------|-------|-----------|
|Poland	|54	|82000	| No |
|Germany|	37	|58000	|Yes|
|United States|	40	|64000|	No|
|Germany |48|	71000	|No|
|United States|	50|	|	Yes|
|Poland	|45|	68000|	Yes|
|Germany	|  |	62000	|No|
|Poland|	58|	89000|	Yes|
|United States	|60	|93000	|No|
|Poland	|47	|77000|	Yes|

__As you can see, we have empty cells, two types of categorical data: country(3 categories) and Purchased(2 categories)__

## Coding Outline:
+ Importing the libraries
+ Importing the dataset
+ Taking care of missing data
+ Encoding categorical data
+ Encoding the Independent Variable
+ Encoding the Dependent Variable
+ Splitting the dataset into the Training set and Test set
+ Feature Scaling

## Data preprocessing codes [To view Code results, click here](https://colab.research.google.com/drive/18MtRgTVlMMmfHGTF_d-mPG2C13YofWaa#scrollTo=TpGqbS4TqkIR)
```python
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
```python
# Importing the dataset
dataset = pd.read_csv('data_c.csv')
X = dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values
print(X)
#print(y)
```
> ## Notes
> The .values attribute helps to extract the data as numpy arrays.
> : is range, [:-1] for columns will exclude the last column since range does not include last values.
+ Class is like set of instructions to follow in order to build an object
+ Object is the instance of the class. The result of excecuting the instructions( class) is the object
+ Method is the tool or function used on object. Performing a particular task in objects.

## Handling missing data [To view Code results, click here](https://colab.research.google.com/drive/18MtRgTVlMMmfHGTF_d-mPG2C13YofWaa#scrollTo=TpGqbS4TqkIR)
+ for large dataset, you may choose to remove the entire row.
+ However, it might not be the best to remove rows because it can affect the quality of the data
+ Use the mean value to replace the missing (nan)
+ To do this we can use the scikit learn library with the SimpleImputer class
+  Use the `transform` method of the `SimpleImputer` class to replace missing data in the specified numerical columns.
+  You can Update the matrix of features by assigning the result of the `transform` method to the correct columns e.g X[:, 1:3].
  
```python
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= pd.NA, strategy='mean')
imputer.fit(X[:, 1:3])
# the transform helps populate the table. In other not to affect entire table
# we update with using exactly the index specified
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

```
## Working with Categorical Data
> Categorical data refers to data that represents categories or groups. It consists of labels or names that describe distinct and non-overlapping groups or classes, rather than numerical values that have inherent mathematical meaning. Categorical data is typically used to represent qualitative characteristics or attributes. Examples: Gender (Male, Female), Colors (Red, Blue, Green), Countries (USA, Canada, India). Answers (Yes, No)

> Key Characteristics of Categorical Data:
Data values are often non-numeric but can be represented numerically (e.g., 0 for Male, 1 for Female).

## Handling Categorical data 
+ Considering the dataset above, we have three categories under country.
+ One can use One-Hot-Encoding to assign these into three columns
+ One-Hot-Encoding introducess a binary vector to each of the countries
+ Example of binary code: say Poland is given __vector (1.0, 0.0, 00), Germany vector(0.0, 1.0, 0.0) USA vector (0.0, 0.0, 1.0).__

### Encoding the Independent Variable
```python
# Encoding
from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
# the transformer takes three items eo-column index, enconder, OneHotEncoder, 
# remainder(helps not to apply to other columns)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
ct.fit_transform(X)

#convert to array
X = np.array(ct.fit_transform(X))

print(X)
```
 [To view Code results, click here](https://colab.research.google.com/drive/18MtRgTVlMMmfHGTF_d-mPG2C13YofWaa#scrollTo=TpGqbS4TqkIR)
 
### Encoding the dependent Variable
```python
# Encoding
# Encoding the dependent variable into 0 and 1, we use LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)
```
 [To view Code results, click here](https://colab.research.google.com/drive/18MtRgTVlMMmfHGTF_d-mPG2C13YofWaa#scrollTo=TpGqbS4TqkIR)

 ## Do you apply feature scaling before splitting dataset or after?
 This is regular question in machine learning. The answer is after splitting the dataset into the training and test set
 > Reason for this is because the test set is a brand new test set, applying the feature scaling before the split means the mean or standard deviation computation would have taken place on the test set which is not supposed to be. Hence, It is done after, so the test set will not be captured in the scaling as it is supposed to be new data with new observation. And, it can cause info leakage on the train set use for the model.

### Spliting Data
```python
# split to train and test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)
```
 [To view Code results, click here](https://colab.research.google.com/drive/18MtRgTVlMMmfHGTF_d-mPG2C13YofWaa#scrollTo=TpGqbS4TqkIR)

 
