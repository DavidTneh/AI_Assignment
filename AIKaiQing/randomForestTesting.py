import tensorflow_decision_forests as tfdf
import pandas as pd

# Load your dataset
df = pd.read_csv('C:/Users/User/OneDrive/Documents/AI_Assignment/train.csv')

# Preprocess your data here
# ...

# Convert the DataFrame to a TensorFlow dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label='log_price', task=tfdf.keras.Task.REGRESSION)

# Create and train the model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Evaluate the model
evaluation = model.evaluate(train_ds)

# Use the model for predictions
predictions = model.predict(train_ds)

# Compile the model with the desired metrics
model.compile(metrics=['mean_squared_error'])

# Evaluate the model on the training dataset
evaluation = model.evaluate(train_ds)

# Output the evaluation results
print(f"Evaluation results: {evaluation}")

# Use the model for predictions
predictions = model.predict(train_ds)

# To view the actual vs predicted values for a few samples, you can do the following:
for i in range(10):  # Display 10 predictions
    print(f"Actual log_price: {df['log_price'].iloc[i]}, Predicted log_price: {predictions[i][0]}")
