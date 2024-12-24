import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
data = pd.read_csv('ClimateDataBasel.csv', header=None)

# Add column names
columns = [
    "Temp_Min", "Temp_Max", "Temp_Mean",
    "Humidity_Min", "Humidity_Max", "Humidity_Mean",
    "Pressure_Min", "Pressure_Max", "Pressure_Mean",
    "Precipitation_Total", "Snowfall_Amount", "Sunshine_Duration",
    "Wind_Gust_Min", "Wind_Gust_Max", "Wind_Gust_Mean",
    "Wind_Speed_Min", "Wind_Speed_Max", "Wind_Speed_Mean"
]
data.columns = columns

# Step 1: Handle Missing Data
def handle_missing_data(data):
    print(f"Missing values detected: {data.isnull().sum().sum()}")
    imputer = SimpleImputer(strategy="mean")
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

data = handle_missing_data(data)

# Step 2: Isolation Forest for Outlier Removal
def isolation_forest_outlier_removal(data, contamination=0.03):
    print("Applying Isolation Forest for outlier detection...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(data)
    print(f"Rows removed due to Isolation Forest outliers: {np.sum(predictions == -1)}")
    return data[predictions == 1].reset_index(drop=True)

data = isolation_forest_outlier_removal(data)

# Step 3: Feature Selection Using Correlation with Heatmap
def feature_selection(data, threshold=0.80):
    print("\nVisualizing Correlation Matrix for Feature Selection...")
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Drop highly correlated features
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    print(f"Features dropped: {to_drop}")
    return data.drop(columns=to_drop)

data = feature_selection(data)

# Step 4: Scaling
# Step 4: Manual Standard Scaling
def manual_standard_scaler(data):
    # Manually implements standard scaling by calculating mean and std deviation
    # for each column (feature).

    means = np.mean(data, axis=0)  # Compute column-wise mean
    stds = np.std(data, axis=0)    # Compute column-wise standard deviation
    
    # Avoid division by zero for any column with zero variance
    stds[stds == 0] = 1.0  
    
    print("Feature-wise Means (Before Scaling):")
    print(means)
    print("\nFeature-wise Standard Deviations (Before Scaling):")
    print(stds)
    
    # Standardize the data
    scaled_data = (data - means) / stds
    
    # Validation to check results
    print("\nFeature-wise Means (After Scaling - Should be close to 0):")
    print(np.mean(scaled_data, axis=0))
    print("\nFeature-wise Standard Deviations (After Scaling - Should be close to 1):")
    print(np.std(scaled_data, axis=0))
    
    return scaled_data
# Apply the manual scaler
scaled_data = manual_standard_scaler(data)

# Step 5: Dimensionality Reduction with t-SNE or UMAP
def apply_umap(data, n_components=2):
    umap_reducer = UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced_data = umap_reducer.fit_transform(data)
    print(f"UMAP Reduction Completed with {n_components} Components")
    return reduced_data

# UMAP applied 
reduced_data = apply_umap(scaled_data)  # Uncomment for UMAP

# Generate the linkage matrix using the 'ward' method
linkage_matrix = linkage(reduced_data, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=5)  # 'p=5' limits the depth to visualize
plt.title("Dendrogram for Agglomerative Clustering")
plt.xlabel("Cluster Index")
plt.ylabel("Distance")
plt.show()

# Step 6: Spectral Clustering Optimization
def optimize_spectral_clustering(data):
    print("\nOptimizing Spectral Clustering...")
    best_score = -1
    best_labels = None
    for clusters in range(2, 10):  # Test multiple cluster numbers
        for gamma in np.linspace(0.5, 2.0, 5):  # Test different gamma values
            similarity_matrix = np.exp(-gamma * squareform(pdist(data))**2)
            laplacian = np.diag(similarity_matrix.sum(axis=1)) - similarity_matrix
            eigvals, eigvecs = eigh(laplacian)
            spectral_embedding = eigvecs[:, 1:clusters + 1]
            labels = KMeans(n_clusters=clusters, random_state=42).fit_predict(spectral_embedding)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
    print(f"Spectral Clustering: Best Silhouette = {best_score:.4f}")
    return best_labels

labels_spectral = optimize_spectral_clustering(reduced_data)

# Step 7: DBSCAN
def dbscan(data, eps_range=(0.1, 1.5), min_samples_range=(5, 30), num_eps_values=100):
    best_silhouette = -1
    best_labels = None
    best_eps = None
    best_min_samples = None
    print("Optimizing DBSCAN Parameters...")
    for min_samples in range(min_samples_range[0], min_samples_range[1], 5):
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = np.sort(distances[:, -1])
        eps_values = np.linspace(eps_range[0], eps_range[1], num_eps_values)
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            valid_labels = labels != -1
            if np.sum(valid_labels) > 1 and len(np.unique(labels[valid_labels])) > 1:
                silhouette = silhouette_score(data[valid_labels], labels[valid_labels])
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_labels = labels
                    best_eps = eps
                    best_min_samples = min_samples
    if best_labels is not None:
        print(f"Best DBSCAN Results: Min Samples = {best_min_samples}, Eps = {best_eps:.4f}, Silhouette Score = {best_silhouette:.4f}")
    else:
        print("DBSCAN failed to produce meaningful clusters.")
    return best_labels

labels_dbscan = dbscan(reduced_data)

# Step 8: Agglomerative Clustering Optimization
def optimize_agglomerative_clustering(data):
    print("\nOptimizing Agglomerative Clustering...")
    best_score = -1
    best_labels = None
    for clusters in range(2, 10):  # Test different numbers of clusters
        for linkage in ['ward', 'average', 'complete']:  # Test different linkage methods
            model = AgglomerativeClustering(n_clusters=clusters, linkage=linkage)
            labels = model.fit_predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
    print(f"Agglomerative Clustering: Best Silhouette = {best_score:.4f}")
    return best_labels

labels_agg = optimize_agglomerative_clustering(reduced_data)

# Step 9: Visualization with Centroids
def plot_clusters_with_centroids(data, labels, title):
    unique_labels = np.unique(labels)
    centroids = np.array([data[labels == label].mean(axis=0) for label in unique_labels if label != -1])
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10, alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title(title)
    plt.colorbar()
    plt.legend()
    plt.show()

# Visualize results
plt.figure()
plot_clusters_with_centroids(reduced_data, labels_spectral, "Optimized Spectral Clustering")
plt.figure()
plot_clusters_with_centroids(reduced_data, labels_dbscan, "DBSCAN Clustering")
plt.figure()
plot_clusters_with_centroids(reduced_data, labels_agg, "Optimized Agglomerative Clustering")
