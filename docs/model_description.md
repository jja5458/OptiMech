# Energy Consumption Simulation Model

## Overview
This simulation model predicts energy consumption in a university commercial building based on various input parameters such as outdoor temperature, indoor setpoint, operating hours, and other relevant features.

## Features Engineered
The following features were engineered to better understand the relationship between system settings and energy consumption:
- **Day of the Week**: To capture weekly patterns in energy usage.
- **Month and Season**: To account for seasonal variation in energy demand.
- **Temperature Difference**: Between indoor and outdoor temperatures.
- **Rolling Statistics**: 7-day rolling mean and standard deviation to capture recent energy consumption trends.

## Model Architecture
The model is a simple **feedforward neural network** with two hidden layers:
- Input layer: 6 input features
- Two hidden layers with 64 and 32 units, respectively
- Output layer: Single value representing energy consumption in kWh.

## Model Training
- **Loss Function**: Mean Squared Error (MSE) was used to train the model.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Epochs**: The model was trained for 1000 epochs.

## Simulation and Use Cases
- The trained model can simulate energy consumption for different configurations of the HVAC system.
- For example, the model can predict energy consumption for varying operating hours, temperature differences, and other configurations.

## Evaluation
The model's performance was evaluated on a test set, and it achieved an MSE loss of approximately [Insert value] on the test data.

## Next Steps
- Implement optimization algorithms to identify the most energy-efficient system configurations.
- Extend the model by incorporating more features or improving the network architecture.
