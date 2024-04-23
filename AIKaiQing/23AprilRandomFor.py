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
selected_columns = ['log_price','property_type', 'room_type', 'bathrooms','amenities', 'bedrooms', 'beds', 'city', 'accommodates', 'cleaning_fee', 'cancellation_policy', 'number_of_reviews', 'instant_bookable', 'host_since', 'host_identity_verified']
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

# Select additional details to display with the actual and predicted prices
additional_details = df_selected.loc[y_test.index, ['city', 'property_type', 'room_type']].reset_index(drop=True)

# Create a DataFrame with actual and predicted prices and additional details
results_df = pd.concat([additional_details, actual_prices, predicted_prices], axis=1)
results_df.columns = ['City', 'Property Type', 'Room Type', 'Actual Price (USD)', 'Predicted Price (USD)']

# Display the first few rows of the results DataFrame
print(results_df.head())

# Assuming 'actual_prices' and 'predicted_prices' are the Series or lists of actual and predicted prices
# Ensure these variables are defined in your code before running this snippet
# Calculate Mean Squared Error (MSE)

mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Calculate R-squared (R2)
r2 = rf.score(X_test, y_test)
print('R-squared (R2):', r2)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(actual_prices, predicted_prices, alpha=0.5, color='blue', label='Actual vs Predicted')

# Plot the line of perfect prediction
plt.plot([min(actual_prices), max(actual_prices)], [min(actual_prices), max(actual_prices)], color='red', linestyle='--', label='Perfect Prediction')

# Label the axes
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Comparison of Actual and Predicted Prices')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# Distribution of Log Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['log_price'], kde=True)
plt.title('Distribution of Log Prices')
plt.xlabel('Log Price')
plt.ylabel('Frequency')
plt.show()

# Frequency of Room Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='room_type')
plt.title('Frequency of Room Types')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show()

# # Geographical Distribution of Properties
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='longitude', y='latitude', hue='log_price', palette='viridis', alpha=0.6)
# plt.title('Geographical Distribution of Properties')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

# Log Price Distribution by Property Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='property_type', y='log_price')
plt.title('Log Price Distribution by Property Type')
plt.xlabel('Property Type')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.ylabel('Log Price')
plt.show()

host_selected = ['host_since','review_scores_rating']

df_selected1 = df.loc[:, host_selected].copy()  # Use .loc to avoid SettingWithCopyWarning

# Convert 'host_since' to datetime
df_selected1['host_since'] = pd.to_datetime(df_selected['host_since'], format='%d-%m-%y')

# Convert 'review_scores_rating' to numeric, coerce errors to NaN
df_selected1['review_scores_rating'] = pd.to_numeric(df_selected['review_scores_rating'], errors='coerce')

# Filter out rows where 'review_scores_rating' is not numeric
df_selected1 = df_selected1[pd.to_numeric(df_selected['review_scores_rating'], errors='coerce').notnull()]

# Fill NaN values with the mean of the column
df_selected1['review_scores_rating'] = df_selected['review_scores_rating'].fillna(df_selected['review_scores_rating'].mean())

# Filter out rows where 'host_since' is null
df_selected1 = df_selected1[df_selected1['host_since'].notnull()]

# Group by 'host_since' and calculate average 'review_scores_rating'
review_scores_over_time = df_selected1.groupby(df_selected1['host_since'].dt.to_period('M')).mean()['review_scores_rating']

# Plot the average review scores rating over time
plt.figure(figsize=(10, 6))
review_scores_over_time.plot()
plt.title('Review Scores Rating Over Time')
plt.xlabel('Time')
plt.ylabel('Average Review Score')
plt.show()

