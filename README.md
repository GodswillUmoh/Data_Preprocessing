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
+ Class is like set of instructions to follow in order to build an object
+ Object is the instance of the class. The result of excecuting the instructions( class) is the object
+ Method is the tool or function used on object. Performing a particular task in objects.
