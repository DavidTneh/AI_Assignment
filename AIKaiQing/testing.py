import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
airbnb_data = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

# Define the selected columns
selected_columns = ['log_price', 'property_type', 'room_type', 'accommodates', 
                    'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 
                    'city', 'host_has_profile_pic', 'host_identity_verified', 
                    'host_response_rate', 'host_since', 'instant_bookable', 
                    'latitude', 'longitude', 'neighbourhood', 'number_of_reviews', 
                    'review_scores_rating', 'bedrooms', 'beds']

# Filter the dataset to include only the selected columns
filtered_data = airbnb_data[selected_columns]

# Drop rows with missing values
filtered_data = filtered_data.dropna()

# Define features (X) and target (y)
X = filtered_data.drop(columns=['log_price'])  # Features
y = filtered_data['log_price']  # Target

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
