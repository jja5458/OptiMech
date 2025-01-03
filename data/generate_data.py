import pandas as pd
import numpy as np

# Adjusting the date range for twelve months of data
date_rng_12months = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

# Generate data for 12 months
data_12months = {
    'Date': date_rng_12months,
    'Indoor Temp (setpoint)': np.random.normal(loc=22, scale=2, size=len(date_rng_12months)),
    'Outdoor Temp': np.random.normal(loc=10, scale=10, size=len(date_rng_12months)),
    'Energy Consumption (kWh)': np.random.normal(loc=1500, scale=300, size=len(date_rng_12months)),
    'Operating Hours': np.random.choice([16, 18, 20], size=len(date_rng_12months)),
    'Refrigerant Pressure (psi)': np.random.normal(loc=30, scale=5, size=len(date_rng_12months)),
    'Hot Water Flow Rate (L/min)': np.random.normal(loc=50, scale=10, size=len(date_rng_12months))
}

# Create DataFrame
df_12months = pd.DataFrame(data_12months)

# Adjustments for seasonality in temperature and energy consumption
df_12months['Outdoor Temp'] += df_12months['Date'].dt.month.map({1: -10, 2: -8, 3: -5, 4: 0, 5: 5, 6: 10, 7: 15, 8: 15, 9: 10, 10: 5, 11: 0, 12: -5})
df_12months['Energy Consumption (kWh)'] *= df_12months['Date'].dt.month.map({1: 1.3, 2: 1.25, 3: 1.1, 4: 0.9, 5: 0.8, 6: 0.75, 7: 0.7, 8: 0.7, 9: 0.8, 10: 0.9, 11: 1.1, 12: 1.2})

df_12months.head()
df_12months.to_csv('energy_data.csv', index=False)

