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

# Get feature importances
feature_importances = gbr.feature_importances_
feature_names = list(numeric_features) + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()

# Sort feature importances
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# Plot the effect of top features on the log price
plt.figure(figsize=(10, 6))
plt.bar(range(10), sorted_importances[:10])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances')
plt.xticks(range(10), sorted_feature_names[:10], rotation=45, ha='right')
plt.tight_layout()
plt.show()
