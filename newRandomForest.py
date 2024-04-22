import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data from the Excel file
df = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

# Select only the specified columns
selected_columns = ['log_price', 'property_type', 'room_type', 'accommodates', 
                    'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 
                    'city', 'host_has_profile_pic', 'host_identity_verified', 
                    'host_response_rate', 'host_since', 'instant_bookable', 
                    'latitude', 'longitude', 'neighbourhood', 'number_of_reviews', 
                    'review_scores_rating', 'bedrooms', 'beds']
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
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# Convert predicted log prices back to actual prices
predicted_prices = pd.Series(predictions).apply(lambda x: round(10 ** x, 2))

# Convert actual log prices back to actual prices for comparison
actual_prices = pd.Series(y_test).apply(lambda x: round(10 ** x, 2))

# Evaluate the model
mse = mean_squared_error(actual_prices, predicted_prices)
print(f"The mean squared error of the Random Forest model is: {mse}")

# Add actual and predicted prices to the DataFrame
results_df = pd.DataFrame({'Actual Price': actual_prices, 'Predicted Price': predicted_prices})
print(results_df.head())  # Display the first few rows
