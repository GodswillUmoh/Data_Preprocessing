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
>
> Data leakage occurs when information from the test set influences the training process.
If scaling is applied before splitting, the test data influences the scaling parameters (mean, standard deviation, min, max, etc.), leading to overly optimistic performance metrics during model evaluation.

### Spliting Data
```python
# split to train and test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)
```
 [To view Code results, click here](https://colab.research.google.com/drive/18MtRgTVlMMmfHGTF_d-mPG2C13YofWaa#scrollTo=TpGqbS4TqkIR)

 ## Feature Scaling
 _Why do we need to do this?_
 > It is to avoid some data dominating in the dataset. Features with larger magnitudes can dominate over features with smaller magnitudes, leading to biased models.
> Feature scaling is a crucial preprocessing step in machine learning, particularly for algorithms that are sensitive to the scale of the data. It ensures that all features contribute equally to the model training process, improving performance and convergence rates.
>
> Feature scaling ensures that all features are on a comparable scale, leading to more accurate results.

 _Which method of Feature scaling should we use? Standardization or Normalization?_
 > Choosing between standardization and normalization depends on the nature of your data and the machine learning algorithm you're using
> _Normalization is use when you have a normal distribution in the data. It should be used When Data Has a Fixed Range._ Normalization scales all features to a range of [0, 1] or [-1, 1], which is useful when the dataset has bounded values or needs to be interpreted as percentages.
>
> _Standardization is used When Data Does Not Have a Defined Range._ It centers the data around a mean of 0 and a standard deviation of 1, which works well for most datasets.
> For Features with Outliers: Standardization is more robust to outliers compared to normalization because it considers the standard deviation rather than min-max ranges.
> __Note: Standardisation works well with different data hence, it is preferred over normalization (works well with some kind of dataset)__

## Dummy Variables
After OnHotEncoder function is applied, the binary numbers assigned to the categorical variable is known as dummy variables. 
_Question: Should feature scaling (e.g Standardisation be applied to dummy variables?_
> The answer is no. This is why: Feature scaling should generally not be applied to dummy variables (also known as one-hot encoded variables) because it distorts their categorical nature and the meaningful interpretation of their values.
> 
> _Dummy Variables Represent Categories:_
Dummy variables represent categories using binary values (e.g., 0 or 1).
These values are not continuous but are used to indicate the presence or absence of a specific category.
Scaling dummy variables changes the binary representation (e.g., scaling 0 and 1 to 0.2 and 0.8), which makes no logical sense for categorical data.

### Applying feature scaling
```python
# standardisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Excluding the index for the dummy variable
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)

print(X_test)
```
 [To view Code results, click here](https://colab.research.google.com/drive/18MtRgTVlMMmfHGTF_d-mPG2C13YofWaa#scrollTo=TpGqbS4TqkIR)

