from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/execute_script', methods=['POST'])
def execute_script():
    try:
        # Load the data
        df = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

        # Select only the specified columns
        selected_columns = ['log_price', 'property_type', 'room_type', 'accommodates','amenities', 
                            'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 
                            'city', 'host_since', 'latitude', 'longitude', 'number_of_reviews', 
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

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, predictions)

        # Convert predicted log prices back to actual prices using the exponential function
        predicted_prices = np.exp(predictions)
        actual_prices = np.exp(y_test)

        # Create a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_prices, predicted_prices, alpha=0.5, color='blue', label='Actual vs Predicted')
        plt.plot([min(actual_prices), max(actual_prices)], [min(actual_prices), max(actual_prices)], color='red', linestyle='--', label='Perfect Prediction')
        plt.xlabel('Actual Price (USD)')
        plt.ylabel('Predicted Price (USD)')
        plt.title('Comparison of Actual and Predicted Prices')
        plt.legend()



        return jsonify({'result': 'Script executed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
