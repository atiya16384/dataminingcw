import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.random_projection import GaussianRandomProjection
from scipy import stats

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

# Select relevant features
selected_features = ['Temperature (Mean)', 'Relative Humidity (Mean)', 'Sea Level Pressure (Mean)', 'Wind Speed (Mean)']
data = data[selected_features]

# Step 1: Standardization
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Step 2: Fourier Transform for Seasonality Removal
def remove_seasonality(column):
    transformed = np.fft.fft(column)
    transformed[int(len(transformed)/2):] = 0  # Remove high frequencies (seasonal)
    return np.fft.ifft(transformed).real

for i, column in enumerate(selected_features):
    data_standardized[:, i] = remove_seasonality(data_standardized[:, i])

# Step 3: Z-score Outlier Removal
z_scores = np.abs(stats.zscore(data_standardized))
data_no_outliers = data_standardized[(z_scores < 3).all(axis=1)]

# Step 4: Polynomial Feature Engineering (New)
poly = PolynomialFeatures(degree=2, interaction_only=True)
data_poly = poly.fit_transform(data_no_outliers)

# Step 5: PCA for Dimensionality Reduction (New)
pca = PCA(n_components=4)  # Reducing the dimensionality
data_pca = pca.fit_transform(data_poly)

# Combine PCA and Gaussian Random Projection for dimensionality reduction
random_projection = GaussianRandomProjection(n_components=2)
data_projected = random_projection.fit_transform(data_pca)

# 5.1 Gaussian Mixture Models (GMM)
print("\n### GMM Clustering ###")
n_components_values = [3, 4, 5, 6, 7, 8]
covariance_types = ['full', 'tied', 'diag', 'spherical']

best_silhouette_gmm = -1
best_n_components_gmm = None
best_covariance_gmm = None

for n_components in n_components_values:
    for cov_type in covariance_types:
        gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type)
        labels_gmm = gmm.fit_predict(data_projected)
        silhouette = silhouette_score(data_projected, labels_gmm)
        
        if silhouette > best_silhouette_gmm:
            best_silhouette_gmm = silhouette
            best_n_components_gmm = n_components
            best_covariance_gmm = cov_type

print(f"Best GMM Silhouette Score: {best_silhouette_gmm}, Components: {best_n_components_gmm}, Covariance Type: {best_covariance_gmm}")
gmm = GaussianMixture(n_components=best_n_components_gmm, covariance_type=best_covariance_gmm)
labels_gmm = gmm.fit_predict(data_projected)
plt.figure()
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=labels_gmm, cmap='plasma', s=50)
plt.title(f'GMM Clustering (Components: {best_n_components_gmm}, Covariance: {best_covariance_gmm})')
plt.show()

# 5.2 Spectral Clustering
print("\n### Spectral Clustering ###")
affinities = ['nearest_neighbors', 'rbf']
n_neighbors_values = [15, 20, 25, 30]
gamma_values = [0.01, 0.1, 1, 5, 10, 20]

best_silhouette_spectral = -1
best_params_spectral = None

for affinity in affinities:
    if affinity == 'nearest_neighbors':
        for n_neighbors in n_neighbors_values:
            spectral = SpectralClustering(n_clusters=4, affinity=affinity, n_neighbors=n_neighbors)
            labels_spectral = spectral.fit_predict(data_projected)
            silhouette = silhouette_score(data_projected, labels_spectral)
            if silhouette > best_silhouette_spectral:
                best_silhouette_spectral = silhouette
                best_params_spectral = (affinity, n_neighbors)
    else:
        for gamma in gamma_values:
            spectral = SpectralClustering(n_clusters=4, affinity=affinity, gamma=gamma)
            labels_spectral = spectral.fit_predict(data_projected)
            silhouette = silhouette_score(data_projected, labels_spectral)
            if silhouette > best_silhouette_spectral:
                best_silhouette_spectral = silhouette
                best_params_spectral = (affinity, gamma)

print(f"Best Spectral Clustering Silhouette Score: {best_silhouette_spectral}, Params: {best_params_spectral}")
if best_params_spectral[0] == 'nearest_neighbors':
    spectral = SpectralClustering(n_clusters=4, affinity=best_params_spectral[0], n_neighbors=best_params_spectral[1])
else:
    spectral = SpectralClustering(n_clusters=4, affinity=best_params_spectral[0], gamma=best_params_spectral[1])
labels_spectral = spectral.fit_predict(data_projected)
plt.figure()
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=labels_spectral, cmap='coolwarm', s=50)
plt.title(f'Spectral Clustering (Params: {best_params_spectral})')
plt.show()

# 5.3 Agglomerative Clustering
print("\n### Agglomerative Clustering ###")
linkages = ['ward', 'complete', 'average', 'single']
n_clusters_values = [3, 4, 5, 6]
best_silhouette_agg = -1
best_linkage = None

for linkage in linkages:
    for n_clusters in n_clusters_values:
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels_agg = agg_clustering.fit_predict(data_projected)
        silhouette = silhouette_score(data_projected, labels_agg)
        
        if silhouette > best_silhouette_agg:
            best_silhouette_agg = silhouette
            best_linkage = (linkage, n_clusters)

print(f"Best Agglomerative Clustering Silhouette Score: {best_silhouette_agg}, Linkage: {best_linkage[0]}, Clusters: {best_linkage[1]}")
agg_clustering = AgglomerativeClustering(n_clusters=best_linkage[1], linkage=best_linkage[0])
labels_agg = agg_clustering.fit_predict(data_projected)
plt.figure()
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=labels_agg, cmap='viridis', s=50)
plt.title(f'Agglomerative Clustering (Best Linkage: {best_linkage[0]}, Clusters: {best_linkage[1]})')
plt.show()
