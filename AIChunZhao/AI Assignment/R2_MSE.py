import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error


airbnb_data = pd.read_csv("C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv")


features = pd.get_dummies(airbnb_data[['property_type', 'room_type', 'amenities','bed_type','cancellation_policy','cleaning_fee',
                                       'city','host_since','instant_bookable','host_identity_verified']])

numerical_features = airbnb_data[['accommodates','bathrooms','latitude','longitude','number_of_reviews',
                                  'review_scores_rating','bedrooms','beds']]

features = pd.concat([numerical_features, features], axis=1)

# Target variable
target = airbnb_data['log_price']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

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
