import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = r'C:\Users\julio\OneDrive - The Pennsylvania State University\Desktop\OptiMech\OptiMech\data\processed_energy_data.csv'
data = pd.read_csv(file_path)

# Drop the 'Date' column as it's not needed for prediction
data = data.drop(columns=['Date'])

# Handle categorical columns (e.g., 'Season', 'Outdoor Temp Bin', 'Day of the Week') using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Handle missing values by filling them with the column mean
data = data.fillna(data.mean())

# Feature selection (excluding the target 'Energy Consumption (kWh)')
features = [col for col in data.columns if col != 'Energy Consumption (kWh)']
X = data[features]  # Features (input data)
y = data['Energy Consumption (kWh)']  # Target variable (energy consumption)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the deep learning model
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input layer with the number of features
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 units and ReLU activation
    layers.Dense(64, activation='relu'),   # Hidden layer with 64 units and ReLU activation
    layers.Dense(32, activation='relu'),   # Hidden layer with 32 units and ReLU activation
    layers.Dense(1)  # Output layer (1 unit for regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Make predictions on the test data
predictions = model.predict(X_test)

# Visualize the predictions vs actual values
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Energy Consumption')
plt.show()

# Optionally, save the trained model for future use
model.save('energy_consumption_model.h5')
