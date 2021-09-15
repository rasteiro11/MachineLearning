# Import libraries
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the decision tree regressor model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualizing the decidion tree regressor results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
