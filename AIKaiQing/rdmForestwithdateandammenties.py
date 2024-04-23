import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data from the CSV file
df = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

# Select the relevant columns
selected_columns = ['log_price', 'property_type', 'room_type', 'accommodates', 
                    'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 
                    'city', 'host_has_profile_pic', 'host_identity_verified', 
                    'host_response_rate', 'host_since', 'instant_bookable', 
                    'latitude', 'longitude', 'neighbourhood', 'number_of_reviews', 
                    'review_scores_rating', 'bedrooms', 'beds', 'amenities']

df_selected = df[selected_columns]

# Convert 'host_since' to datetime format
df_selected.loc[:, 'host_since'] = pd.to_datetime(df_selected['host_since'], errors='coerce', format='%d-%m-%y')

# Check if 'host_since' column is in datetime format
if pd.api.types.is_datetime64_any_dtype(df_selected['host_since']):
    # Calculate 'days_since_host' if 'host_since' is in datetime format
    df_selected.loc[:, 'days_since_host'] = (pd.to_datetime('today') - df_selected['host_since']).dt.days
else:
    print("The 'host_since' column is not in datetime format.")

# Drop the original 'host_since' column without chain indexing
df_selected.drop(columns=['host_since'], inplace=True)

# Update the numeric and non-numeric columns
numeric_columns = df_selected.select_dtypes(include=['number']).columns
non_numeric_columns = df_selected.select_dtypes(exclude=['number']).columns

# Update the ColumnTransformer to handle the new columns
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

# Initialize and train the model
rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# Convert predicted log prices back to actual prices using the exponential function
predicted_prices = pd.Series(predictions).apply(lambda x: round(np.exp(x), 2))

# Convert actual log prices back to actual prices for comparison
actual_prices = pd.Series(y_test).apply(lambda x: round(np.exp(x), 2)).reset_index(drop=True)

# Select additional details to display with the actual and predicted prices
additional_details = df_selected.loc[y_test.index, ['city', 'property_type', 'room_type']].reset_index(drop=True)

# Create a DataFrame with actual and predicted prices and additional details
results_df = pd.concat([additional_details, actual_prices, predicted_prices], axis=1)
results_df.columns = ['City', 'Property Type', 'Room Type', 'Actual Price (USD)', 'Predicted Price (USD)']

# Display the first few rows of the results DataFrame
print(results_df.head())

# Calculate Mean Squared Error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, predictions)
r2 = rf.score(X_test, y_test)

# Display evaluation metrics
print('Mean Squared Error:', mse)
print('R-squared (R2):', r2)
