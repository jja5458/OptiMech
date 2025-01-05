import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EnergyConsumptionModel(nn.Module):
    """
    Neural network model to simulate energy consumption based on input features.
    """
    def __init__(self, input_dim):
        super(EnergyConsumptionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)         # Second hidden layer
        self.fc3 = nn.Linear(32, 1)          # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_and_preprocess_data(file_path):
    """
    Load the dataset, perform necessary preprocessing steps, and return X and y.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Convert 'Date' to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Select input features and target variable
    X = data[['Day of the Week', 'Month', 'Temp Difference', 'Operating Hours', 'Rolling Mean 7', 'Rolling Std 7']]
    y = data['Energy Consumption (kWh)']
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    return X_tensor, y_tensor, scaler

def train_model(X_train, y_train, input_dim, epochs=1000, lr=0.001):
    """
    Train the energy consumption prediction model.
    """
    model = EnergyConsumptionModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        y_pred = model(X_train)
        
        # Calculate loss
        loss = loss_fn(y_pred, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    """
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
    
    test_loss = nn.MSELoss()(y_pred_test, y_test)
    print(f'Test Loss (MSE): {test_loss.item():.4f}')

def simulate_energy_consumption(model, scaler, input_features):
    """
    Simulate the energy consumption for a given configuration.
    """
    input_scaled = scaler.transform([input_features])  # Apply scaling
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        energy_pred = model(input_tensor)
    
    return energy_pred.item()

if __name__ == "__main__":
    # Load and preprocess the data
    file_path = r'C:\Users\julio\OneDrive - The Pennsylvania State University\Desktop\OptiMech\OptiMech\data\processed_energy_data.csv'
    X_tensor, y_tensor, scaler = load_and_preprocess_data(file_path)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train, X_train.shape[1])
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Simulate energy consumption with new configuration
    new_config = [3, 6, 10, 20, 1400, 200]  # Example values
    predicted_energy = simulate_energy_consumption(model, scaler, new_config)
    print(f'Predicted Energy Consumption for new configuration: {predicted_energy:.2f} kWh')
