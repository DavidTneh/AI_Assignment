from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error


# Read the CSV file
airbnb_data = pd.read_csv("C:\\Download\\AI Assignment\\train.csv")

# Select only the specified columns
selected_columns = ['log_price', 'property_type', 'room_type', 'accommodates', 'amenities', 
                    'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 
                    'city', 'host_since', 'latitude', 'longitude', 'number_of_reviews', 
                    'review_scores_rating', 'bedrooms', 'beds', 'instant_bookable', 'host_identity_verified']

df_selected = airbnb_data[selected_columns]

# Target variable
target = df_selected['log_price']

# Features
features = df_selected.drop(columns=['log_price'])

# Preprocessing: OneHotEncoding for categorical variables and scaling for numerical variables
numeric_features = features.select_dtypes(include=['number']).columns
categorical_features = features.select_dtypes(exclude=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Preprocess the data
X_processed = preprocessor.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, target, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)

# Make predictions
y_pred = gbr.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R-squared:", r2)
print("Mean Squared Error:", mse)



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
