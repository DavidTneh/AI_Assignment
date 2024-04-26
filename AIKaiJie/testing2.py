import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Load data
data = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

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

# Get feature names after preprocessing
feature_names = preprocessor.transformers_[0][1].get_feature_names(non_numeric_columns).tolist() + \
                preprocessor.transformers_[1][1].get_feature_names(numeric_columns).tolist() + \
                numeric_columns.tolist()

# Extract coefficients and corresponding feature names
coefficients = model.coef_
feature_coefficients = pd.Series(coefficients, index=feature_names)

# Sort feature coefficients by magnitude
sorted_coefficients = feature_coefficients.abs().sort_values(ascending=False)

# Select top features (you can adjust the number of features to display)
top_features = sorted_coefficients.head(10)

# Plot the effect of top features on the log price
plt.figure(figsize=(10, 6))
top_features.plot(kind='bar')
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Effect of Top Features on Log Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
