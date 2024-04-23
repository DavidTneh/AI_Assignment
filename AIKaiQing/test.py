import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Load data
data = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

# Select relevant features and target variable
features = ['property_type', 'room_type', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude', 'city', 'accommodates', '', 'cancellation_policy']
target = 'log_price'

# Extract features and target variable
X = data[features]
y = data[target]

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the entire dataset
all_predictions = model.predict(X)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Calculate R-squared (R2)
r2 = model.score(X_test, y_test)
print('R-squared (R2):', r2)

# Scatter plot of actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Accuracy of price prediction')
plt.grid(True)
plt.show()