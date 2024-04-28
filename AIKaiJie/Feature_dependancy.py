import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('C:/Users/ASUS/Downloads/Artificial_intelligence/Assignment/train.csv')

# Select relevant features and target variable
features = ['property_type', 'room_type', 'bathrooms', 'amenities', 'bedrooms', 'beds', 
            'city', 'accommodates', 'cleaning_fee', 'cancellation_policy', 
            'number_of_reviews', 'review_scores_rating', 'latitude', 'longitude', 
            'instant_bookable', 'host_since', 'host_identity_verified']
target = 'log_price'

# Drop non-numeric columns before computing correlation coefficients
numeric_data = data[features + [target]].select_dtypes(include=['number'])

# Compute correlation coefficients between features and target variable
correlation_matrix = numeric_data.corr().abs()
correlation_with_target = correlation_matrix[target].drop(target)  # Drop target's correlation with itself
correlation_with_target = correlation_with_target.sort_values(ascending=False)

# Plot the dependency of features on the target variable with x and y axes switched
plt.figure(figsize=(12, 8))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, color='blue')
plt.xlabel('Feature')
plt.ylabel('Absolute Correlation Coefficient')
plt.title('Dependency of Log Price on Features')
plt.grid(axis='y')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.show()



