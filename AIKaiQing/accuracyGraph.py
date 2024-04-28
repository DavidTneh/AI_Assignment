import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Load the data from the CSV file
df = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

# Select only the specified columns
selected_columns = ['log_price', 'property_type', 'room_type', 'accommodates','amenities', 
                    'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 
                    'city',
                     'host_since', 
                    'latitude', 'longitude', 'number_of_reviews', 
                    'review_scores_rating', 'bedrooms', 'beds', 'instant_bookable','host_identity_verified']


df_selected = df[selected_columns]

# Identify numeric and non-numeric columns
numeric_columns = df_selected.select_dtypes(include=['number']).columns
non_numeric_columns = df_selected.select_dtypes(exclude=['number']).columns

# Create a ColumnTransformer to apply OneHotEncoding to non-numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), non_numeric_columns)
    ],
    remainder='passthrough'
)

# Apply the preprocessing to the feature matrix
X = df_selected.drop('log_price', axis=1)
y = df_selected['log_price']
X_processed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# Convert predicted log prices back to actual prices using the exponential function
predicted_prices = pd.Series(predictions).apply(lambda x: round(np.exp(x), 2))

# Convert actual log prices back to actual prices for comparison
actual_prices = pd.Series(y_test).apply(lambda x: round(np.exp(x), 2)).reset_index(drop=True)

mse = mean_squared_error(y_test, predictions)
r2 = rf.score(X_test, y_test)
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
