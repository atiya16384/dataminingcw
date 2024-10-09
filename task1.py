import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Load the dataset
columns = [
    'Temperature (Min)', 'Temperature (Max)', 'Temperature (Mean)',
    'Relative Humidity (Min)', 'Relative Humidity (Max)', 'Relative Humidity (Mean)',
    'Sea Level Pressure (Min)', 'Sea Level Pressure (Max)', 'Sea Level Pressure (Mean)',
    'Precipitation Total', 'Snowfall Amount', 'Sunshine Duration',
    'Wind Gust (Min)', 'Wind Gust (Max)', 'Wind Gust (Mean)',
    'Wind Speed (Min)', 'Wind Speed (Max)', 'Wind Speed (Mean)'
]

data = pd.read_csv('ClimateDataBasel.csv', names=columns)

# 1. Handle missing data (Imputation - from the list)
imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean
data_imputed = imputer.fit_transform(data)

# 2. Fourier Transform for Seasonality Removal (Advanced Research Technique)
def remove_seasonality(data_column):
    fft_values = np.fft.fft(data_column)
    # Zero out low-frequency components to remove seasonality
    fft_values[1:4] = 0
    return np.fft.ifft(fft_values).real

# Apply Fourier Transform to specific columns to remove seasonality
for column in ['Temperature (Mean)', 'Relative Humidity (Mean)', 'Sea Level Pressure (Mean)', 'Wind Speed (Mean)']:
    data_imputed[:, columns.index(column)] = remove_seasonality(data_imputed[:, columns.index(column)])

# 3. Matrix Factorization (SVD) for Dimensionality Reduction (Advanced Research Technique)
# Apply TruncatedSVD to reduce the dimensionality of the dataset
svd = TruncatedSVD(n_components=10)
data_svd = svd.fit_transform(data_imputed)

# 4. Standardization (from the list)
# Standardize the data to ensure all features have zero mean and unit variance
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_svd)

# Visualization of Preprocessing Results
plt.figure(figsize=(10, 6))

# Plot each of the standardized temperature variables separately
plt.subplot(2, 1, 1)
plt.plot(data_standardized[:, 0], label='Temperature (Min)')
plt.title('Temperature (Min) - Standardized')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data_standardized[:, 1], label='Temperature (Max)', color='orange')
plt.title('Temperature (Max) - Standardized')
plt.legend()

plt.tight_layout()
plt.show()

# Visualization of more features (Sea Level Pressure, Wind Speed)
plt.figure(figsize=(10, 6))

# Plot Sea Level Pressure and Wind Speed
plt.plot(data_standardized[:, columns.index('Sea Level Pressure (Mean)')], label='Sea Level Pressure (Mean)', color='green')
plt.plot(data_standardized[:, columns.index('Wind Speed (Mean)')], label='Wind Speed (Mean)', color='red')

plt.title('Sea Level Pressure and Wind Speed - Standardized')
plt.legend()
plt.show()

# Print out the first few rows of the preprocessed data for inspection
print(pd.DataFrame(data_standardized, columns=[f"PC{i+1}" for i in range(data_standardized.shape[1])]).head())
