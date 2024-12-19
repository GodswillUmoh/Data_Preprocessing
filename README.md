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
