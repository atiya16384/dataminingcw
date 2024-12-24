# Import necessary libraries
import os
import warnings
import numpy as np
print("NumPy Version:", np.__version__)
import torch
from torchvision import transforms, datasets, models
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import umap
import matplotlib.pyplot as plt
import time
from PIL import Image, UnidentifiedImageError

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)  # Fix: Correct API usage
# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress Truncated File Read warning
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress division warnings in metrics
# Constants
BATCH_SIZE = 32
NUM_COMPONENTS = 2  # For dimensionality reduction

# Visualize an image
# Constants
image_path = 'kagglecatsanddogs_3367a/PetImages/Cat/0.jpg'

# Load the image as a PIL image
try:
    original_image = Image.open(image_path).convert("RGB")  # Ensure RGB format
except FileNotFoundError:
    print("The specified image file was not found.")
    exit()

# Step 1: Validate and Preprocess Datasets
# Validate dataset to remove corrupted or non-image files
def validate_dataset(base_dir):
    for category in ["Cat", "Dog"]:
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):  # Skip if it's not a directory
            continue
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for valid image extensions
                print(f"Non-image file removed: {file}")
                os.remove(file_path)  # Remove invalid files
                continue
            try:
                img = Image.open(file_path)
                img.verify()
            except (UnidentifiedImageError, IOError):
                print(f"Corrupted file removed: {file}")
                os.remove(file_path)

# Define transforms for preprocessing
transform = transforms.Compose([
    # Resize to 256x256
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Randomly crop to 224x224
    transforms.RandomResizedCrop(224),
    # Normalize with ImageNet statistics
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Apply transformations
transformed_image = transform(original_image)

# Extract pixel intensities for visualization
# Original Image Pixel Intensities
original_pixels = np.array(original_image).flatten()

# Transformed Image Pixel Intensities (After Normalization)
transformed_pixels = transformed_image.view(-1).numpy()

# Visualization - Histograms of Pixel Values
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot histogram of original image pixel values
ax[0].hist(original_pixels, bins=50, color='blue', alpha=0.7)
ax[0].set_title("Original Image Pixel Distribution")
ax[0].set_xlabel("Pixel Intensity (0-255)")
ax[0].set_ylabel("Frequency")

# Plot histogram of transformed image pixel values
ax[1].hist(transformed_pixels, bins=50, color='green', alpha=0.7)
ax[1].set_title("Transformed Image Pixel Distribution")
ax[1].set_xlabel("Pixel Intensity (Normalized)")
ax[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
# Load datasets
print("Loading datasets...")
cats_dogs_dataset = datasets.ImageFolder(root="kagglecatsanddogs_3367a/PetImages", transform=transform)
oxford_pets_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=transform)

# load the datasets
cats_dogs_dataloader = DataLoader(cats_dogs_dataset, batch_size=BATCH_SIZE, shuffle=True)
oxford_pets_dataloader = DataLoader(oxford_pets_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 2: Feature Extraction with ResNet and VGG16
# Define models for ResNet and VGG16
resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
vgg16_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Replace the final fully connected layer with an Identity layer
# to extract features from the second-to-last layer
resnet_model.fc = torch.nn.Identity()
vgg16_model.classifier[6] = torch.nn.Identity()

# Add a Fine-Tunable Linear Layer
class FineTuneDNN(torch.nn.Module):
    # Define a simple feedforward neural network
    def __init__(self, input_dim, num_classes):
        super(FineTuneDNN, self).__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Function to extract features and fine-tune a linear layer
def extract_and_finetune(model, dataloader, num_classes):
    # Extract features from the second-to-last layer
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs).flatten(start_dim=1)
            features.append(outputs)
            labels.append(targets)
            # Early stopping for demonstration purposes
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # Train Linear Layer on extracted features
    print("Training Linear Layer...")
    linear_layer = FineTuneDNN(features.shape[1], num_classes)
    # Use CrossEntropyLoss and Adam optimizer
    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # Train for 5 epochs - update as needed (50 would be ideal)
    epochs = 5

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Convert to PyTorch tensors
    for epoch in range(epochs):
        linear_layer.train()
        # Convert to PyTorch tensors
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.long)
        # Zero the gradients
        optimizer.zero_grad()
        outputs = linear_layer(inputs)
        # Calculate loss
        loss = criterion(outputs, targets)
        loss.backward()
        # Update weights
        optimizer.step()

        # Calculate accuracy
        # Get the class with the highest probability
        _, predicted = torch.max(outputs, 1) 
        correct = (predicted == targets).sum().item()
        accuracy = 100 * correct / targets.size(0)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    return features, labels

# Extract features for ResNet and VGG16
# Fine-tune the linear layer for classification
print("Extracting features and fine-tuning linear layer with ResNet...")
features_resnet_cd, labels_resnet_cd = extract_and_finetune(resnet_model, cats_dogs_dataloader, num_classes=2)
features_resnet_pets, labels_resnet_pets = extract_and_finetune(resnet_model, oxford_pets_dataloader, num_classes=37)

print("Extracting features and fine-tuning linear layer with VGG16...")
features_vgg16_cd, labels_vgg16_cd = extract_and_finetune(vgg16_model, cats_dogs_dataloader, num_classes=2)
features_vgg16_pets, labels_vgg16_pets = extract_and_finetune(vgg16_model, oxford_pets_dataloader, num_classes=37)

# Custom Standard Scaler
def standard_scaler(features):
    # Compute column-wise mean and standard deviation
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero for columns with zero variance
    std[std == 0] = 1.0
    return (features - mean) / std

# Standardize the extracted features
features_resnet_cd = standard_scaler(features_resnet_cd)
features_resnet_pets = standard_scaler(features_resnet_pets)
features_vgg16_cd = standard_scaler(features_vgg16_cd)
features_vgg16_pets = standard_scaler(features_vgg16_pets)

# Step 3: Dimensionality Reduction
def reduce_dimensions(features, n_components=NUM_COMPONENTS):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(features)

print("Reducing dimensions...")
# Reduce dimensions for all datasets
reduced_features_resnet_cd = reduce_dimensions(features_resnet_cd)
reduced_features_resnet_pets = reduce_dimensions(features_resnet_pets)
reduced_features_vgg16_cd = reduce_dimensions(features_vgg16_cd)
reduced_features_vgg16_pets = reduce_dimensions(features_vgg16_pets)

# UMAP Visualization
def visualize_umap(features, labels, title):
    # Plot UMAP projections
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="Spectral", s=5)
    plt.colorbar(boundaries=np.arange(len(set(labels)) + 1) - 0.5).set_ticks(np.arange(len(set(labels))))
    plt.title(title)
    plt.show()

print("Visualizing UMAP projections...")
visualize_umap(reduced_features_resnet_cd, labels_resnet_cd, "UMAP Projection - ResNet (Cats vs Dogs)")
visualize_umap(reduced_features_resnet_pets, labels_resnet_pets, "UMAP Projection - ResNet (Oxford-IIIT Pets)")
visualize_umap(reduced_features_vgg16_cd, labels_vgg16_cd, "UMAP Projection - VGG16 (Cats vs Dogs)")
visualize_umap(reduced_features_vgg16_pets, labels_vgg16_pets, "UMAP Projection - VGG16 (Oxford-IIIT Pets)")

# Step 4: Clustering
def purity_score(true_labels, cluster_labels):
    # Compute confusion matrix
    contingency_matrix = confusion_matrix(true_labels, cluster_labels)
    # Return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def k_means_clustering(features, true_labels, n_clusters):
    # Initialize centroids randomly
    np.random.seed(42)
    centroids = features[np.random.choice(features.shape[0], n_clusters, replace=False)]

    # K-means clustering
    for _ in range(100):
        # Compute distances and assign clusters
        distances = np.linalg.norm(features[:, None] - centroids, axis=2)
        # Assign clusters based on minimum distance
        cluster_labels = np.argmin(distances, axis=1)
        # Update centroids
        new_centroids = np.array([features[cluster_labels == k].mean(axis=0) for k in range(n_clusters)])
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    # Compute evaluation metrics
    silhouette_avg = silhouette_score(features, cluster_labels)
    davies_bouldin = davies_bouldin_score(features, cluster_labels)
    purity = purity_score(true_labels, cluster_labels)

    return cluster_labels, silhouette_avg, davies_bouldin, purity

# Perform K-means clustering
print("Clustering datasets...")
for name, features, labels in [
    ("ResNet Cats vs Dogs", reduced_features_resnet_cd, labels_resnet_cd),
    ("ResNet Oxford-IIIT Pets", reduced_features_resnet_pets, labels_resnet_pets),
    ("VGG16 Cats vs Dogs", reduced_features_vgg16_cd, labels_vgg16_cd),
    ("VGG16 Oxford-IIIT Pets", reduced_features_vgg16_pets, labels_vgg16_pets),
]:
    cluster_labels, silhouette_avg, davies_bouldin, purity = k_means_clustering(features, labels, n_clusters=2)
    print(f"--- {name} Clustering ---")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"Purity Score: {purity}")

# Step 5: Classification
def classify_features(features, labels, method="logistic"):
    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize the classification model
    if method == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif method == "knn":
        model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    elif method == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif method == "lda":
        model = LDA()
    else:
        raise ValueError("Invalid classification method")

    # Train the model and evaluate performance
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Make predictions and evaluate performance
    predictions = model.predict(X_test)
    metrics = classification_report(y_test, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f"\n--- {method.upper()} Classification Results ---")
    print(f"Training Time: {training_time:.4f} seconds")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(conf_matrix)

    return metrics, conf_matrix

# Classify features using different methods
print("Classifying features...")
for model_name, features, labels in [
    ("ResNet Cats vs Dogs", reduced_features_resnet_cd, labels_resnet_cd),
    ("ResNet Oxford-IIIT Pets", reduced_features_resnet_pets, labels_resnet_pets),
    ("VGG16 Cats vs Dogs", reduced_features_vgg16_cd, labels_vgg16_cd),
    ("VGG16 Oxford-IIIT Pets", reduced_features_vgg16_pets, labels_vgg16_pets),
]:
    # Classify using different methods
    for method in ["logistic", "knn", "random_forest", "lda"]:
        print(f"--- {method.upper()} Classification ({model_name}) ---")
        classify_features(features, labels, method)
