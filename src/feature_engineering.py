import pandas as pd

# Load the dataset
file_path = r'C:\Users\julio\OneDrive - The Pennsylvania State University\Desktop\OptiMech\OptiMech\data\energy_data.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Create time-based features
data['Day of the Week'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else ('Spring' if x in [3, 4, 5] else ('Summer' if x in [6, 7, 8] else 'Fall')))

# Create weather-related features
data['Temp Difference'] = data['Indoor Temp (setpoint)'] - data['Outdoor Temp']
data['Outdoor Temp Bin'] = data['Outdoor Temp'].apply(lambda x: 'Cold' if x < 5 else ('Moderate' if x <= 25 else 'Hot'))

# Create lag features
data['Energy Consumption Lag 1'] = data['Energy Consumption (kWh)'].shift(1)

# Create rolling statistics
data['Rolling Mean 7'] = data['Energy Consumption (kWh)'].rolling(window=7).mean()
data['Rolling Std 7'] = data['Energy Consumption (kWh)'].rolling(window=7).std()

# Interaction features
data['Energy vs Temp'] = data['Energy Consumption (kWh)'] * data['Temp Difference']

# Handle missing values and export
data = data.dropna()
data.to_csv(r'C:\Users\julio\OneDrive - The Pennsylvania State University\Desktop\OptiMech\OptiMech\data\processed_energy_data.csv', index=False)

# View the first few rows of the processed data
print(data.head())
