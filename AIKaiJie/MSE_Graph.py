import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('C:/Users/ASUS/Downloads/Artificial_intelligence/Assignment/train.csv')

# Select relevant features and target variable
features = ['property_type', 'room_type', 'bathrooms', 'amenities', 'bedrooms', 'beds', 
            'city', 'accommodates', 'cleaning_fee', 'cancellation_policy', 
            'number_of_reviews', 'review_scores_rating', 'latitude', 'longitude', 
            'instant_bookable', 'host_since', 'host_identity_verified']
target = 'log_price'

# Extract features and target variable
X = data[features]
y = data[target]

# Identify numeric and non-numeric columns
numeric_columns = X.select_dtypes(include=['number']).columns
non_numeric_columns = X.select_dtypes(exclude=['number']).columns

# Create a ColumnTransformer to apply transformations to features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), non_numeric_columns),
        ('poly', PolynomialFeatures(degree=2, include_bias=False), numeric_columns),
        ('scale', StandardScaler(), numeric_columns)
    ],
    remainder='passthrough'
)

# Apply the preprocessing to the feature matrix
X_processed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train the Ridge regression model with regularization
model = Ridge(alpha=1.0)  # You can adjust the regularization strength (alpha)
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Calculate R-squared (R2)
r2 = r2_score(y_test, predictions)
print('R-squared (R2):', r2)

# Plotting R-squared and MSE
plt.figure(figsize=(10, 6))

# Plot R-squared
plt.bar(['R-squared'], [r2], color='blue', alpha=0.5, label='R-squared')

# Plot MSE
plt.bar(['Mean Squared Error'], [mse], color='red', alpha=0.5, label='Mean Squared Error')

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Evaluation Metrics')
plt.legend()
plt.grid(True)
plt.show()
