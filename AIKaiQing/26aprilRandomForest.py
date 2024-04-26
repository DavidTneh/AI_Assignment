import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from flask import Flask, render_template
from pdpbox import pdp, info_plots

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

features_to_plot = ['property_type', 'room_type', 'accommodates']
for feature_name in features_to_plot:
    pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_test, model_features=X.columns, feature=feature_name)
    pdp.pdp_plot(pdp_dist, feature_name)
    plt.title(f'Partial Dependence Plot for {feature_name}')
    plt.show()

# Create interaction PDPs for selected pairs of features
interactions = [('property_type', 'room_type'), ('property_type', 'accommodates'), ('room_type', 'accommodates')]
for interaction_pair in interactions:
    pdp_interaction = pdp.pdp_interact(model=rf, dataset=X_test, model_features=X.columns, features=interaction_pair)
    pdp.pdp_interact_plot(pdp_interaction, interaction_pair, plot_type='grid', x_quantile=True)
    plt.title(f'Interaction PDP for {interaction_pair}')
    plt.show()