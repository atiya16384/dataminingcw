import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL

# Load the dataset (update with your file path)
columns = [
    'Temperature (Min)', 'Temperature (Max)', 'Temperature (Mean)',
    'Relative Humidity (Min)', 'Relative Humidity (Max)', 'Relative Humidity (Mean)',
    'Sea Level Pressure (Min)', 'Sea Level Pressure (Max)', 'Sea Level Pressure (Mean)',
    'Precipitation Total', 'Snowfall Amount', 'Sunshine Duration',
    'Wind Gust (Min)', 'Wind Gust (Max)', 'Wind Gust (Mean)',
    'Wind Speed (Min)', 'Wind Speed (Max)', 'Wind Speed (Mean)'
]

data = pd.read_csv('ClimateDataBasel.csv', names=columns)

# 1. STL Decomposition (Time Series Decomposition)
def stl_decomposition(data, column):
    stl = STL(data[column], seasonal=13)
    result = stl.fit()
    result.plot()
    plt.title(f'STL Decomposition - {column}')
    plt.show()
    return result.trend, result.seasonal, result.resid

# Apply STL decomposition to all relevant columns
for feature in ['Temperature (Mean)', 'Relative Humidity (Mean)', 'Sea Level Pressure (Mean)', 'Wind Speed (Mean)']:
    trend, seasonal, resid = stl_decomposition(data, feature)
    data[f'{feature}_trend'] = trend
    data[f'{feature}_seasonal'] = seasonal
    data[f'{feature}_resid'] = resid

# Using the trend component for clustering to remove seasonal effects
data_for_clustering = data[[f'{feature}_trend' for feature in ['Temperature (Mean)', 'Relative Humidity (Mean)', 'Sea Level Pressure (Mean)', 'Wind Speed (Mean)']]]

# 2. t-SNE (Nonlinear Dimensionality Reduction)
# Fill any missing values (if present) with zeros for simplicity
data_for_clustering.fillna(0, inplace=True)

# Apply t-SNE for dimensionality reduction to 2 dimensions
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
data_tsne = tsne.fit_transform(data_for_clustering)

# Convert t-SNE result to a DataFrame
tsne_df = pd.DataFrame(data_tsne, columns=['TSNE_1', 'TSNE_2'])

# Plot t-SNE result
plt.scatter(tsne_df['TSNE_1'], tsne_df['TSNE_2'])
plt.title('t-SNE Result for Basel Climate Data')
plt.show()

# 3. Cluster-Based Feature Engineering
# Cluster temperature-related and wind-related features separately
temperature_features = data[['Temperature (Min)', 'Temperature (Max)', 'Temperature (Mean)']]
wind_features = data[['Wind Speed (Min)', 'Wind Speed (Max)', 'Wind Speed (Mean)']]

# Apply KMeans clustering on temperature-related features
kmeans_temp = KMeans(n_clusters=3, random_state=42).fit(temperature_features)
data['Temp_Cluster'] = kmeans_temp.labels_

# Apply KMeans clustering on wind-related features
kmeans_wind = KMeans(n_clusters=3, random_state=42).fit(wind_features)
data['Wind_Cluster'] = kmeans_wind.labels_

# Use the new meta-features for further analysis
meta_features = data[['Temp_Cluster', 'Wind_Cluster']]

# 4. Robust Scaling (Normalization)
# Select all relevant features for scaling
scaling_features = data[
    ['Temperature (Min)', 'Temperature (Max)', 'Temperature (Mean)',
     'Relative Humidity (Min)', 'Relative Humidity (Max)', 'Relative Humidity (Mean)',
     'Sea Level Pressure (Min)', 'Sea Level Pressure (Max)', 'Sea Level Pressure (Mean)',
     'Precipitation Total', 'Snowfall Amount', 'Sunshine Duration',
     'Wind Gust (Min)', 'Wind Gust (Max)', 'Wind Gust (Mean)',
     'Wind Speed (Min)', 'Wind Speed (Max)', 'Wind Speed (Mean)']
]

# Apply RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(scaling_features)

# Convert the scaled data back to a DataFrame for easier use
scaled_df = pd.DataFrame(scaled_data, columns=scaling_features.columns)

# Print out the first few rows of the preprocessed data
print(scaled_df.head())
