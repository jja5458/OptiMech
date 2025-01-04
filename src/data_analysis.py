import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
file_path = 'C:\\Users\\julio\\OneDrive - The Pennsylvania State University\\Desktop\\OptiMech\\OptiMech\\data\\energy_data.csv'
data = pd.read_csv(file_path)

# Basic preprocessing
# Convert 'Date' to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Check for missing values
print(data.isnull().sum())

# Describe the data to see the distributions
print(data.describe())

# Set plot style
sns.set(style="whitegrid")

# Plotting Energy Consumption over time
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Energy Consumption (kWh)'], label='Energy Consumption')
plt.title('Daily Energy Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 8))
corr_matrix = data.drop('Date', axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Parameters')
plt.show()

