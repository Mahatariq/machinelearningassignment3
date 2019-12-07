# Simple Linear Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('aids.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,2:3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'pink')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('years vs deaths (Training set)')
plt.xlabel('Years ')
plt.ylabel(' no of deaths')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'brown')
plt.title('years vs no of deaths (Test set)')
plt.xlabel('Years ')
plt.ylabel('no of deaths')
plt.show()
print(regressor.predict([[1992]]))