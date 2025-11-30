# ============================================================================
# Section 1: Environment Setup and Data Loading
# ============================================================================

# Configure environment for GPU detection (must be before TensorFlow import)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Import core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from pathlib import Path

# Image processing libraries
from PIL import Image

# Sklearn preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

# Sklearn model selection
from sklearn.model_selection import train_test_split

# Sklearn metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    silhouette_score
)

# Models for classification/clustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Deep Learning for feature extraction
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Clustering Analysis
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Deep Learning for model building and evaluation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Set random seed for reproducibility
RANDOM_STATE = 67  # we're aware 42 is the industry standard, but we haven't read Hitchhiker's Guide to the Galaxy and 67 is funnier
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("Libraries imported successfully!")

# GPU Detection and Configuration
print("GPU DETECTION")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {gpus}")

# Define the data directory
DATA_DIR = Path('.data')

# Initialize lists to store image paths and labels
image_paths = []
labels = []

# Scan the data directory for category folders
print("Scanning dataset directory...")

# Get all category folders
category_folders = [f for f in DATA_DIR.iterdir() if f.is_dir()]
category_folders = sorted(category_folders)

print(f"Found {len(category_folders)} categories")
print("\nScanning images in each category...")

# Iterate through each category folder
for category_folder in category_folders:
    category_name = category_folder.name
    
    # Find all image files in the category folder
    # Support common image formats: jpg, jpeg, png, bmp
    image_files = list(category_folder.glob('*.jpg')) + \
                  list(category_folder.glob('*.jpeg')) + \
                  list(category_folder.glob('*.png')) + \
                  list(category_folder.glob('*.bmp'))
    
    # Add to our lists
    for img_path in image_files:
        image_paths.append(str(img_path))
        labels.append(category_name)

# Create a DataFrame
df = pd.DataFrame({
    'image_path': image_paths,
    'category': labels
})

# Display basic information
print("\n" + "."*50)
print("Dataset loaded successfully!")
print("."*50)
print(f"\nDataset shape: {df.shape}")
print(f"Total number of images: {df.shape[0]:,}")
print(f"Number of categories: {df['category'].nunique()}")

print("\n" + "."*50)
print("Category Distribution:")
print("."*50)
category_counts = df['category'].value_counts().sort_index()
print(f"\nImages per category (first 10):")
print(category_counts.head(10))

print(f"\nStatistics:")
print(f"  Mean images per category: {category_counts.mean():.1f}")
print(f"  Median images per category: {category_counts.median():.1f}")
print(f"  Min images per category: {category_counts.min()}")
print(f"  Max images per category: {category_counts.max()}")

print("\n" + "."*50)
print("First few rows of the dataset:")
print(df.head(10))

# Sample a few images to check dimensions and formats
print("Analyzing image properties...")

sample_size = min(100, len(df))
sample_indices = np.random.choice(len(df), size=sample_size, replace=False)

widths = []
heights = []
formats = []
corrupted_images = []

for idx in sample_indices:
    img_path = df.iloc[idx]['image_path']
    try:
        img = Image.open(img_path)
        widths.append(img.width)
        heights.append(img.height)
        formats.append(img.format)
        img.close()
    except Exception as e:
        corrupted_images.append(img_path)
        print(f"Warning: Could not read {img_path}: {e}")

if len(corrupted_images) > 0:
    print(f"\nFound {len(corrupted_images)} corrupted images")
else:
    print("\nAll sampled images are valid")

print(f"\nImage Dimensions (from {sample_size} samples):")
print(f"Width  - Min: {min(widths)}px, Max: {max(widths)}px, Mean: {np.mean(widths):.1f}px")
print(f"Height - Min: {min(heights)}px, Max: {max(heights)}px, Mean: {np.mean(heights):.1f}px")
print(f"\nImage Formats: {set(formats)}")

# Analyze category distribution
print("\n" + "."*50)
print("Category Distribution Analysis:")
print("."*50)

category_counts = df['category'].value_counts()
print(f"\nTop 10 most common categories:")
print(category_counts.head(10))

print(f"\nTop 10 least common categories:")
print(category_counts.tail(10))

# Check for class imbalance
imbalance_ratio = category_counts.max() / category_counts.min()
print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio > 3:
    print("Significant class imbalance detected")
else:
    print("Relatively balanced dataset")

# Store key information for later use
print("\n" + "."*50)
print("Dataset Summary:")
print("."*50)
print(f"Total Images: {len(df):,}")
print(f"Total Categories: {df['category'].nunique()}")
print(f"Average Images per Category: {len(df) / df['category'].nunique():.1f}")

# ============================================================================
# Section 2: Exploratory Data Analysis (EDA) and Feature Extraction
# ============================================================================

# Visualize sample images from different categories
n_categories_to_show = 8
n_images_per_category = 3

# Select random categories
random_categories = np.random.choice(df['category'].unique(), 
                                     size=min(n_categories_to_show, df['category'].nunique()), 
                                     replace=False)

# Create figure
fig, axes = plt.subplots(n_categories_to_show, n_images_per_category, 
                         figsize=(15, 2.5*n_categories_to_show))

if n_categories_to_show == 1:
    axes = axes.reshape(1, -1)

print("Displaying sample images from random categories...")
print("."*50)

for i, category in enumerate(random_categories):
    # Get images from this category
    category_df = df[df['category'] == category]
    
    # Sample random images
    sampled_images = category_df.sample(n=min(n_images_per_category, len(category_df)))
    
    for j, (idx, row) in enumerate(sampled_images.iterrows()):
        img_path = row['image_path']
        
        try:
            img = Image.open(img_path)
            
            # Display image
            if n_categories_to_show > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
                
            ax.imshow(img)
            ax.axis('off')
            
            if j == 0:  # Add category label to first image in row
                ax.set_title(f"{category}\n({len(category_df)} images)", 
                           fontsize=10, fontweight='bold')
            
            img.close()
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

plt.tight_layout()
plt.savefig('images/sample_images_grid.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional: Show category name statistics
print("\n" + "."*50)
print("Category Name Analysis:")
print("."*50)
print(f"Shortest category name: '{min(df['category'].unique(), key=len)}' ({len(min(df['category'].unique(), key=len))} chars)")
print(f"Longest category name: '{max(df['category'].unique(), key=len)}' ({len(max(df['category'].unique(), key=len))} chars)")
print(f"\nAll categories ({len(df['category'].unique())} total):")
print(sorted(df['category'].unique()))

# Analyze category distribution in detail
print("Category Distribution Analysis")

category_counts = df['category'].value_counts()

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Top 20 categories bar chart
ax1 = fig.add_subplot(gs[0, 0])
top_20 = category_counts.head(20)
ax1.barh(range(len(top_20)), top_20.values, color='steelblue')
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20.index, fontsize=9)
ax1.set_xlabel('Number of Images', fontsize=11)
ax1.set_title('Top 20 Categories by Image Count', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(top_20.values):
    ax1.text(v + 5, i, str(v), va='center', fontsize=8)

# 2. Bottom 20 categories bar chart
ax2 = fig.add_subplot(gs[0, 1])
bottom_20 = category_counts.tail(20).sort_values(ascending=False)
ax2.barh(range(len(bottom_20)), bottom_20.values, color='coral')
ax2.set_yticks(range(len(bottom_20)))
ax2.set_yticklabels(bottom_20.index, fontsize=9)
ax2.set_xlabel('Number of Images', fontsize=11)
ax2.set_title('Bottom 20 Categories by Image Count', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(bottom_20.values):
    ax2.text(v + 1, i, str(v), va='center', fontsize=8)

# 3. Histogram of category sizes
ax3 = fig.add_subplot(gs[1, :])
ax3.hist(category_counts.values, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax3.axvline(category_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {category_counts.mean():.1f}')
ax3.axvline(category_counts.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {category_counts.median():.1f}')
ax3.set_xlabel('Number of Images per Category', fontsize=11)
ax3.set_ylabel('Frequency (Number of Categories)', fontsize=11)
ax3.set_title('Distribution of Images Across Categories', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 4. Box plot
ax4 = fig.add_subplot(gs[2, 0])
ax4.boxplot(category_counts.values, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax4.set_ylabel('Number of Images', fontsize=11)
ax4.set_title('Box Plot of Category Sizes', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Statistics summary text
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
stats_text = f"""
CATEGORY DISTRIBUTION STATISTICS

Total Categories: {len(category_counts)}
Total Images: {category_counts.sum():,}

Images per Category:
  • Mean:     {category_counts.mean():.2f}
  • Median:   {category_counts.median():.2f}
  • Std Dev:  {category_counts.std():.2f}
  • Min:      {category_counts.min()}
  • Max:      {category_counts.max()}
  • Q1:       {category_counts.quantile(0.25):.2f}
  • Q3:       {category_counts.quantile(0.75):.2f}

Class Imbalance Ratio: {category_counts.max() / category_counts.min():.2f}x

Most Common: {category_counts.index[0]} ({category_counts.values[0]} images)
Least Common: {category_counts.index[-1]} ({category_counts.values[-1]} images)
"""
ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('images/category_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature Extraction using MobileNetV2
print("Initializing Feature Extraction")

# Load pre-trained MobileNetV2 model without top classification layer, this is pur feature extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # Remove classification head
    weights='imagenet',
    pooling='avg'  # Global average pooling to get fixed-size feature vector
)

print(f"MobileNetV2 loaded successfully")
print(f"Feature vector dimension: {base_model.output_shape[1]}")

# Function to preprocess and extract features from an image
def extract_features(img_path, target_size=(224, 224)):
    """Extract features from a single image using MobileNetV2"""
    try:
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Extract features
        features = base_model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Extract features for all images (with progress tracking)
print("\nExtracting features from all images...")
print("This may take several minutes...")
print("."*70)

features_list = []
valid_indices = []
failed_images = []

# Process images in batches for efficiency
batch_size = 32
n_batches = int(np.ceil(len(df) / batch_size))

for batch_idx in range(n_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(df))
    
    batch_images = []
    batch_indices = []
    
    # Load batch of images
    for idx in range(start_idx, end_idx):
        img_path = df.iloc[idx]['image_path']
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            batch_images.append(img_array)
            batch_indices.append(idx)
        except Exception as e:
            failed_images.append(img_path)
            continue
    
    if len(batch_images) > 0:
        # Preprocess batch
        batch_images = np.array(batch_images)
        batch_images = tf.keras.applications.mobilenet_v2.preprocess_input(batch_images)
        
        # Extract features for batch
        batch_features = base_model.predict(batch_images, verbose=0)
        
        # Store features and indices
        for i, features in enumerate(batch_features):
            features_list.append(features)
            valid_indices.append(batch_indices[i])
    
    # Progress update
    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
        progress = ((batch_idx + 1) / n_batches) * 100
        print(f"Progress: {progress:.1f}% ({batch_idx + 1}/{n_batches} batches) - "
              f"{len(features_list)} images processed")

# Convert to numpy array
features_array = np.array(features_list)

# Create cleaned dataframe with only successfully processed images
df_features = df.iloc[valid_indices].copy().reset_index(drop=True)


print("\n Feature Extraction Complete!")

print(f"Successfully extracted features from: {len(features_list):,} images")
print(f"Failed to process: {len(failed_images)} images")
print(f"Feature matrix shape: {features_array.shape}")
print(f"{features_array.shape[0]:,} samples")
print(f"{features_array.shape[1]} features per image")
print(f"\nMemory usage: {features_array.nbytes / (1024**2):.2f} MB")

# Analyze feature distributions
print("Feature Distribution Analysis")

# Basic statistics
feature_means = features_array.mean(axis=0)
feature_stds = features_array.std(axis=0)
feature_mins = features_array.min(axis=0)
feature_maxs = features_array.max(axis=0)

print(f"\nFeature Statistics Summary:")
print(f"  Mean of means: {feature_means.mean():.4f}")
print(f"  Mean of stds:  {feature_stds.mean():.4f}")
print(f"  Global min:    {feature_mins.min():.4f}")
print(f"  Global max:    {feature_maxs.max():.4f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Distribution of feature means
axes[0, 0].hist(feature_means, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Feature Mean Value', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].set_title('Distribution of Feature Means', fontsize=11, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# 2. Distribution of feature standard deviations
axes[0, 1].hist(feature_stds, bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Feature Std Dev', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].set_title('Distribution of Feature Std Deviations', fontsize=11, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3. Distribution of feature ranges
feature_ranges = feature_maxs - feature_mins
axes[0, 2].hist(feature_ranges, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Feature Range (Max - Min)', fontsize=10)
axes[0, 2].set_ylabel('Frequency', fontsize=10)
axes[0, 2].set_title('Distribution of Feature Ranges', fontsize=11, fontweight='bold')
axes[0, 2].grid(alpha=0.3)

# 4. Sample feature values across images (first 100 features)
n_features_to_plot = min(100, features_array.shape[1])
im = axes[1, 0].imshow(features_array[:200, :n_features_to_plot].T, 
                       aspect='auto', cmap='viridis', interpolation='nearest')
axes[1, 0].set_xlabel('Image Index', fontsize=10)
axes[1, 0].set_ylabel(f'Feature Index (first {n_features_to_plot})', fontsize=10)
axes[1, 0].set_title('Feature Value Heatmap (Sample)', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=axes[1, 0])

# 5. Feature correlation sample (random subset to avoid memory issues)
n_features_corr = min(50, features_array.shape[1])
random_feature_indices = np.random.choice(features_array.shape[1], n_features_corr, replace=False)
feature_subset = features_array[:, random_feature_indices]
correlation_matrix = np.corrcoef(feature_subset.T)

im2 = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
axes[1, 1].set_xlabel('Feature Index', fontsize=10)
axes[1, 1].set_ylabel('Feature Index', fontsize=10)
axes[1, 1].set_title(f'Feature Correlation Matrix ({n_features_corr} random features)', 
                     fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=axes[1, 1])

# 6. Distribution of a few sample features
sample_features_idx = np.random.choice(features_array.shape[1], 5, replace=False)
for i, feat_idx in enumerate(sample_features_idx):
    axes[1, 2].hist(features_array[:, feat_idx], bins=30, alpha=0.5, 
                    label=f'Feature {feat_idx}', edgecolor='black')
axes[1, 2].set_xlabel('Feature Value', fontsize=10)
axes[1, 2].set_ylabel('Frequency', fontsize=10)
axes[1, 2].set_title('Sample Feature Distributions', fontsize=11, fontweight='bold')
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/feature_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Check for zero-variance features
zero_var_features = np.where(feature_stds < 1e-10)[0]
print(f"\nZero-variance features: {len(zero_var_features)}")
if len(zero_var_features) > 0:
    print(f"  These features have no variation and could be removed")

# Check sparsity
sparsity = (features_array == 0).sum() / features_array.size * 100
print(f"\nFeature sparsity: {sparsity:.2f}% (percentage of zero values)")

# Dimensionality Reduction for Visualization
print("Applying Dimensionality Reduction Techniques")

# For visualization, we'll use a subset if dataset is very large
max_samples_for_viz = 5000
if len(features_array) > max_samples_for_viz:
    print(f"\nDataset has {len(features_array)} samples - using stratified sample of {max_samples_for_viz} for visualization")
    # Stratified sampling to maintain category proportions
    from sklearn.model_selection import train_test_split
    _, X_viz, _, y_viz = train_test_split(
        features_array, 
        df_features['category'],
        train_size=(len(features_array) - max_samples_for_viz),
        stratify=df_features['category'],
        random_state=RANDOM_STATE
    )
else:
    X_viz = features_array
    y_viz = df_features['category']

print(f"Visualization sample size: {len(X_viz)} images")

# 1. PCA - Principal Component Analysis
print("\n" + "."*70)
print("1. Applying PCA...")
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)

X_pca_2d = pca_2d.fit_transform(X_viz)
X_pca_3d = pca_3d.fit_transform(X_viz)

print(f"PCA 2D: Explained variance: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%")
print(f"PCA 3D: Explained variance: {pca_3d.explained_variance_ratio_.sum()*100:.2f}%")

# 2. t-SNE - T-distributed Stochastic Neighbor Embedding
print("\n" + "."*70)
print("2. Applying t-SNE (this may take a few minutes)...")

# Use PCA preprocessing for t-SNE (recommended practice)
pca_50 = PCA(n_components=50, random_state=RANDOM_STATE)
X_pca_50 = pca_50.fit_transform(X_viz)

tsne_2d = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
X_tsne_2d = tsne_2d.fit_transform(X_pca_50)

print(f"t-SNE 2D completed")

# Create visualizations
print("\n Creating visualizations...")

# Encode categories as numbers for coloring
le = LabelEncoder()
y_viz_encoded = le.fit_transform(y_viz)

# Create a colormap
n_categories = len(np.unique(y_viz_encoded))
colors = plt.cm.tab20(np.linspace(0, 1, min(n_categories, 20)))

# Create figure with subplots
fig = plt.figure(figsize=(20, 10))

# 1. PCA 2D
ax1 = fig.add_subplot(2, 2, 1)
scatter1 = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                       c=y_viz_encoded, cmap='tab20', 
                       s=20, alpha=0.6, edgecolors='none')
ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax1.set_title('PCA - 2D Projection', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# 2. PCA 3D
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
scatter2 = ax2.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                       c=y_viz_encoded, cmap='tab20',
                       s=20, alpha=0.6, edgecolors='none')
ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
ax2.set_title('PCA - 3D Projection', fontsize=13, fontweight='bold')

# 3. t-SNE 2D
ax3 = fig.add_subplot(2, 2, 3)
scatter3 = ax3.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1],
                       c=y_viz_encoded, cmap='tab20',
                       s=20, alpha=0.6, edgecolors='none')
ax3.set_xlabel('t-SNE Dimension 1', fontsize=11)
ax3.set_ylabel('t-SNE Dimension 2', fontsize=11)
ax3.set_title('t-SNE - 2D Projection', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3)

# 4. Explained variance plot for PCA
ax4 = fig.add_subplot(2, 2, 4)
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_viz)
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

ax4.plot(range(1, min(51, len(cumsum_variance)+1)), 
         cumsum_variance[:50], 
         'b-', linewidth=2, label='Cumulative Variance')
ax4.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
ax4.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
ax4.axvline(x=n_components_90, color='r', linestyle=':', alpha=0.5)
ax4.axvline(x=n_components_95, color='g', linestyle=':', alpha=0.5)
ax4.set_xlabel('Number of Components', fontsize=11)
ax4.set_ylabel('Cumulative Explained Variance', fontsize=11)
ax4.set_title(f'PCA Variance Explained (90%={n_components_90}, 95%={n_components_95} components)', 
              fontsize=11, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.legend(fontsize=9)
ax4.set_xlim(0, 50)

plt.tight_layout()
plt.savefig('images/dimensionality_reduction_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "."*70)
print("Dimensionality Reduction Summary:")
print("."*70)
print(f"PCA Analysis:")
print(f"Components for 90% variance: {n_components_90}")
print(f"Components for 95% variance: {n_components_95}")
print(f"First 2 components capture: {pca_2d.explained_variance_ratio_.sum()*100:.2f}% of variance")
print(f"First 3 components capture: {pca_3d.explained_variance_ratio_.sum()*100:.2f}% of variance")

# ============================================================================
# Section 3: Data Preprocessing and Dataset Splitting
# ============================================================================

# Feature Scaling and Label Encoding
print("Preprocessing Features and Labels")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_features['category'])

print(f"\nLabel Encoding:")
print(f"  Number of classes: {len(label_encoder.classes_)}")
print(f"  Classes: {label_encoder.classes_[:10]}..." if len(label_encoder.classes_) > 10 else f"  Classes: {label_encoder.classes_}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_array)

print(f"\nFeature Scaling:")
print(f"  Original features - Mean: {features_array.mean():.4f}, Std: {features_array.std():.4f}")
print(f"  Scaled features   - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
print(f"\nPreprocessing complete!")

# Create stratified train/validation/test splits
print("Creating Train/Validation/Test Splits")

# First split: separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.15,
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# Second split: separate validation set from remaining data (15% of total = ~17.6% of temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176,  # This gives us 15% of the original data
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"\nDataset Splits:")
print(f"  Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"  Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"  Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"  Total:          {len(X_scaled):,} samples")

# Verify class distribution
print(f"\nClass Distribution Verification:")
print(f"  Original:   {np.bincount(y_encoded).min()} to {np.bincount(y_encoded).max()} samples per class")
print(f"  Training:   {np.bincount(y_train).min()} to {np.bincount(y_train).max()} samples per class")
print(f"  Validation: {np.bincount(y_val).min()} to {np.bincount(y_val).max()} samples per class")
print(f"  Test:       {np.bincount(y_test).min()} to {np.bincount(y_test).max()} samples per class")

# ============================================================================
# Section 4: Clustering Analysis
# ============================================================================

# Number of clusters = number of categories
n_clusters = len(label_encoder.classes_)
print(f"Number of clusters: {n_clusters}\n")

# K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

km_silhouette = silhouette_score(X_scaled, kmeans_labels)
km_nmi = normalized_mutual_info_score(y_encoded, kmeans_labels)
km_ari = adjusted_rand_score(y_encoded, kmeans_labels)

# Hierarchical Clustering
hier = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
hier_labels = hier.fit_predict(X_scaled)

hier_silhouette = silhouette_score(X_scaled, hier_labels)
hier_nmi = normalized_mutual_info_score(y_encoded, hier_labels)
hier_ari = adjusted_rand_score(y_encoded, hier_labels)


# Visualization

# Compute PCA and t-SNE for full dataset
pca_2d_full = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d_full = pca_2d_full.fit_transform(X_scaled)

pca_50_full = PCA(n_components=50, random_state=RANDOM_STATE)
X_pca_50_full = pca_50_full.fit_transform(X_scaled)
tsne_2d_full = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
X_tsne_2d_full = tsne_2d_full.fit_transform(X_pca_50_full)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# K-Means - PCA
axes[0, 0].scatter(X_pca_2d_full[:, 0], X_pca_2d_full[:, 1], c=kmeans_labels, cmap='tab20', s=10, alpha=0.6)
axes[0, 0].set_title('K-Means - PCA', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')

# K-Means - t-SNE
axes[0, 1].scatter(X_tsne_2d_full[:, 0], X_tsne_2d_full[:, 1], c=kmeans_labels, cmap='tab20', s=10, alpha=0.6)
axes[0, 1].set_title('K-Means - t-SNE', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('t-SNE 1')
axes[0, 1].set_ylabel('t-SNE 2')

# True categories - t-SNE
axes[0, 2].scatter(X_tsne_2d_full[:, 0], X_tsne_2d_full[:, 1], c=y_encoded, cmap='tab20', s=10, alpha=0.6)
axes[0, 2].set_title('True Categories - t-SNE', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('t-SNE 1')
axes[0, 2].set_ylabel('t-SNE 2')

# Hierarchical - PCA
axes[1, 0].scatter(X_pca_2d_full[:, 0], X_pca_2d_full[:, 1], c=hier_labels, cmap='tab20', s=10, alpha=0.6)
axes[1, 0].set_title('Hierarchical - PCA', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

# Hierarchical - t-SNE
axes[1, 1].scatter(X_tsne_2d_full[:, 0], X_tsne_2d_full[:, 1], c=hier_labels, cmap='tab20', s=10, alpha=0.6)
axes[1, 1].set_title('Hierarchical - t-SNE', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('t-SNE 1')
axes[1, 1].set_ylabel('t-SNE 2')

# Metrics comparison
axes[1, 2].axis('off')
metrics_text = f"""
CLUSTERING METRICS

K-Means:
  Silhouette: {km_silhouette:.4f}
  NMI:        {km_nmi:.4f}
  ARI:        {km_ari:.4f}

Hierarchical:
  Silhouette: {hier_silhouette:.4f}
  NMI:        {hier_nmi:.4f}
  ARI:        {hier_ari:.4f}
"""
axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('images/clustering_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Section 5: Deep Learning - ANN
# ============================================================================

# --- Constants ---
FEATURE_DIMENSION = 1280
NB_CATEGORIES = 233     # Number of folders in dataset
EPOCHS = 50 
BATCH_SIZE = 64
PATIENCE = 10 
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)

print(f"Feature Dimension: {FEATURE_DIMENSION}")
print(f"Number of Categories: {NB_CATEGORIES}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Convert integer labels to one-hot encoding for Softmax output layer
y_train_one_hot = to_categorical(y_train, NB_CATEGORIES)
y_val_one_hot = to_categorical(y_val, NB_CATEGORIES)
y_test_one_hot = to_categorical(y_test, NB_CATEGORIES)

print(f"\nOne-hot encoded label shapes:")
print(f"y_train_one_hot: {y_train_one_hot.shape}")
print(f"y_val_one_hot: {y_val_one_hot.shape}")
print(f"y_test_one_hot: {y_test_one_hot.shape}")

# MODEL DEFINITION: Deep Dense Network (PatternMind Architecture)
def build_patternmind_net(input_dim, classes):
    """
    Builds a deep dense network for high-dimensional feature classification.
    
    Architecture Design:
    - Layer 1 (512 units): High capacity to capture complex feature interactions
    - Layer 2 (256 units): Medium capacity for intermediate representations
    - Layer 3 (128 units): Lower capacity to consolidate learned patterns
    - Output Layer (softmax): Multi-class probability distribution
    
    Regularization:
    - BatchNormalization: Stabilizes training by normalizing layer inputs
    - Dropout (0.5, 0.3): Prevents overfitting by randomly dropping connections
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential(name="PatternMind_ANN")
    
    # Layer 1: High capacity hidden layer
    model.add(Dense(units=512, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())  # Stabilize training
    model.add(Dropout(0.5))          # Prevent overfitting

    # Layer 2: Medium capacity hidden layer
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Layer 3: Lower capacity layer
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer (Softmax for multi-class probability distribution)
    model.add(Dense(units=classes, activation='softmax', name='output_classification'))
    
    return model

# Build the model
patternmind_model = build_patternmind_net(FEATURE_DIMENSION, NB_CATEGORIES)

# Print Model Summary
patternmind_model.summary()

# COMPILATION AND TRAINING SETUP
patternmind_model.compile(
    optimizer=OPTIMIZER,
    loss='categorical_crossentropy',  # Standard loss for multi-class classification
    metrics=['accuracy']
)

# Early Stopping callback to halt training when validation loss plateaus
es_callback = EarlyStopping(
    monitor='val_loss',        # Monitor validation loss
    patience=PATIENCE,         # Wait 10 epochs before stopping
    restore_best_weights=True, # Restore weights from best epoch
    verbose=1
)

print("Model Architecture Defined and Compiled.")
print(f"\nTraining Configuration:")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss Function: Categorical Crossentropy")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Early Stopping Patience: {PATIENCE} epochs")

# Train the model
print("Starting model training...\n")

history = patternmind_model.fit(
    X_train, y_train_one_hot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_one_hot),
    callbacks=[es_callback],
    verbose=1
)

print("\n" + "."*60)
print("Training Complete!")
print("."*60)

# Visualize training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot training & validation accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot training & validation loss
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/ann_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\nFinal ANN Training Metrics:")
print(f"  Training Accuracy:   {final_train_acc:.4f}")
print(f"  Training Loss:       {final_train_loss:.4f}")
print(f"  Validation Accuracy: {final_val_acc:.4f}")
print(f"  Validation Loss:     {final_val_loss:.4f}")

# ============================================================================
# Section 5: Deep Learning - CNN
# ============================================================================

# Reconstruct train/val/test splits matching ANN splits
print("\nCreating stratified train/val/test splits...")
temp_indices = list(range(len(features_array)))
train_temp_indices, test_temp_indices = train_test_split(
    temp_indices, test_size=0.15, stratify=y_encoded, random_state=RANDOM_STATE
)
train_final_indices, val_final_indices = train_test_split(
    train_temp_indices, test_size=0.176, stratify=y_encoded[train_temp_indices], random_state=RANDOM_STATE
)

# Create DataFrames for each split
df_train_cnn = df_features.iloc[train_final_indices].copy()
df_val_cnn = df_features.iloc[val_final_indices].copy()
df_test_cnn = df_features.iloc[test_temp_indices].copy()

print(f"   Train: {len(df_train_cnn):,} samples")
print(f"   Val:   {len(df_val_cnn):,} samples")
print(f"   Test:  {len(df_test_cnn):,} samples")

# Image parameters 
IMG_HEIGHT = 128
IMG_WIDTH = 128
CNN_BATCH_SIZE = 16 # Half of ANN batch size due to higher memory needs

print(f"\nImage configuration:")
print(f"   Input shape: ({IMG_HEIGHT}, {IMG_WIDTH}, 3)")
print(f"   Batch size: {CNN_BATCH_SIZE}")

# DATA AUGMENTATION for training set (Regularization)
print("\nSetting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0,1]
    rotation_range=20,           # Random rotation ±20°
    width_shift_range=0.2,       # Horizontal shift ±20%
    height_shift_range=0.2,      # Vertical shift ±20%
    horizontal_flip=True,        # Random horizontal flip
    zoom_range=0.2,              # Random zoom ±20%
    fill_mode='nearest'          # Fill pixels after transforms
)

# Validation/test set: only rescaling (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

print("Training: Rotation, shifts, flips, zoom enabled")
print("Validation/Test: Only rescaling")

# Get exact class ordering from label encoder 
class_list = list(label_encoder.classes_)
print(f"\nUsing {len(class_list)} classes from label encoder")
print(f"   First 5 classes: {class_list[:5]}")

# Create data generators
print("\nCreating data generators...")
train_generator = train_datagen.flow_from_dataframe(
    df_train_cnn,
    x_col='image_path',
    y_col='category',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=CNN_BATCH_SIZE,
    class_mode='categorical',
    classes=class_list,  
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_dataframe(
    df_val_cnn,
    x_col='image_path',
    y_col='category',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=CNN_BATCH_SIZE,
    class_mode='categorical',
    classes=class_list,
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    df_test_cnn,
    x_col='image_path',
    y_col='category',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=CNN_BATCH_SIZE,
    class_mode='categorical',
    classes=class_list,
    shuffle=False
)

from tensorflow.keras import models, layers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN with learning rate scheduling to escape local minima.
    Key improvements:
    - He Normal initialization for better gradient flow
    - Deeper architecture with more filters
    - Strategic dropout placement
    - Better regularization
    """
    model = models.Sequential(name='CNN')
    
    # Block 1: Initial feature extraction (32 filters)
    model.add(layers.Conv2D(32, (3, 3), padding='same', 
                           kernel_initializer=HeNormal(),
                           input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                           kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2: Mid-level features (64 filters)
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                           kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same',
                           kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Block 3: High-level features (128 filters)
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                           kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',
                           kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Block 4: Deep features (256 filters)
    model.add(layers.Conv2D(256, (3, 3), padding='same',
                           kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same',
                           kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    
    # Global Average Pooling (better than flatten for CNNs)
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense layers
    model.add(layers.Dense(512, kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256, kernel_initializer=HeNormal()))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile with slightly higher initial learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build the improved model
cnn_model = build_cnn_model(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes=NB_CATEGORIES
)

# Print model summary
cnn_model.summary()

# Setup callbacks with learning rate reduction
early_stopping_cnn = EarlyStopping(
    monitor='val_loss',
    patience=15,  # Increased patience for LR reduction to work
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_cnn.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ReduceLROnPlateau - KEY for escaping local minima
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce LR by half
    patience=5,  # Wait 5 epochs before reducing
    min_lr=1e-7,  # Don't go below this
    verbose=1
)

import datetime
# Setup TensorBoard logging
log_dir = "logs/cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cnn = TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=False,
    write_images=False,
    update_freq='epoch',
    profile_batch=0
)

# Clear memory
gc.collect()
tf.keras.backend.clear_session()

print("\nStarting CNN training...")
print("-" * 70)

# Train with all callbacks including ReduceLROnPlateau
cnn_history = cnn_model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[
        early_stopping_cnn,
        model_checkpoint,
        reduce_lr,  # This is KEY for escaping local minima
        tensorboard_cnn
    ],
    verbose=1
)

# Print final metrics
final_train_acc_cnn = cnn_history.history['accuracy'][-1]
final_val_acc_cnn = cnn_history.history['val_accuracy'][-1]
final_train_loss_cnn = cnn_history.history['loss'][-1]
final_val_loss_cnn = cnn_history.history['val_loss'][-1]

print(f"\nFinal CNN Metrics:")
print(f"  Training Accuracy:   {final_train_acc_cnn:.4f} ({final_train_acc_cnn*100:.2f}%)")
print(f"  Validation Accuracy: {final_val_acc_cnn:.4f} ({final_val_acc_cnn*100:.2f}%)")
print(f"  Training Loss:       {final_train_loss_cnn:.4f}")
print(f"  Validation Loss:     {final_val_loss_cnn:.4f}")

# Compare ANN and CNN training histories
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Get the minimum number of epochs between both models (in case early stopping differed)
min_epochs_ann = len(history.history['loss'])
min_epochs_cnn = len(cnn_history.history['loss'])

# 1. Training Loss Comparison
axes[0, 0].plot(range(1, min_epochs_ann + 1), history.history['loss'], 
                label='ANN Training Loss', linewidth=2, marker='o', markersize=4)
axes[0, 0].plot(range(1, min_epochs_cnn + 1), cnn_history.history['loss'], 
                label='CNN Training Loss', linewidth=2, marker='s', markersize=4)
axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 2. Validation Loss Comparison
axes[0, 1].plot(range(1, min_epochs_ann + 1), history.history['val_loss'], 
                label='ANN Validation Loss', linewidth=2, marker='o', markersize=4)
axes[0, 1].plot(range(1, min_epochs_cnn + 1), cnn_history.history['val_loss'], 
                label='CNN Validation Loss', linewidth=2, marker='s', markersize=4)
axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# 3. Training Accuracy Comparison
axes[1, 0].plot(range(1, min_epochs_ann + 1), history.history['accuracy'], 
                label='ANN Training Accuracy', linewidth=2, marker='o', markersize=4)
axes[1, 0].plot(range(1, min_epochs_cnn + 1), cnn_history.history['accuracy'], 
                label='CNN Training Accuracy', linewidth=2, marker='s', markersize=4)
axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Accuracy', fontsize=12)
axes[1, 0].legend(fontsize=11, loc='lower right')
axes[1, 0].grid(True, alpha=0.3)

# 4. Validation Accuracy Comparison
axes[1, 1].plot(range(1, min_epochs_ann + 1), history.history['val_accuracy'], 
                label='ANN Validation Accuracy', linewidth=2, marker='o', markersize=4)
axes[1, 1].plot(range(1, min_epochs_cnn + 1), cnn_history.history['val_accuracy'], 
                label='CNN Validation Accuracy', linewidth=2, marker='s', markersize=4)
axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Accuracy', fontsize=12)
axes[1, 1].legend(fontsize=11, loc='lower right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/ann_vs_cnn_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print comparison summary
print("\n" + "."*70)
print("MODEL COMPARISON SUMMARY")
print("."*70)

print("\n--- ANN (with MobileNetV2 features) ---")
print(f"  Total Epochs Trained:    {min_epochs_ann}")
print(f"  Final Training Loss:     {history.history['loss'][-1]:.4f}")
print(f"  Final Validation Loss:   {history.history['val_loss'][-1]:.4f}")
print(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final Val Accuracy:      {history.history['val_accuracy'][-1]:.4f}")
print(f"  Best Validation Loss:    {min(history.history['val_loss']):.4f} (Epoch {np.argmin(history.history['val_loss']) + 1})")
print(f"  Best Val Accuracy:       {max(history.history['val_accuracy']):.4f} (Epoch {np.argmax(history.history['val_accuracy']) + 1})")

print("\n--- CNN (from raw images) ---")
print(f"  Total Epochs Trained:    {min_epochs_cnn}")
print(f"  Final Training Loss:     {cnn_history.history['loss'][-1]:.4f}")
print(f"  Final Validation Loss:   {cnn_history.history['val_loss'][-1]:.4f}")
print(f"  Final Training Accuracy: {cnn_history.history['accuracy'][-1]:.4f}")
print(f"  Final Val Accuracy:      {cnn_history.history['val_accuracy'][-1]:.4f}")
print(f"  Best Validation Loss:    {min(cnn_history.history['val_loss']):.4f} (Epoch {np.argmin(cnn_history.history['val_loss']) + 1})")
print(f"  Best Val Accuracy:       {max(cnn_history.history['val_accuracy']):.4f} (Epoch {np.argmax(cnn_history.history['val_accuracy']) + 1})")

print("\n" + "."*70)
print("INSIGHTS")
print("."*70)

# Determine which model performed better
ann_best_val_acc = max(history.history['val_accuracy'])
cnn_best_val_acc = max(cnn_history.history['val_accuracy'])
ann_best_val_loss = min(history.history['val_loss'])
cnn_best_val_loss = min(cnn_history.history['val_loss'])

if ann_best_val_acc > cnn_best_val_acc:
    print(f"ANN achieved higher validation accuracy ({ann_best_val_acc:.4f} vs {cnn_best_val_acc:.4f})")
else:
    print(f"CNN achieved higher validation accuracy ({cnn_best_val_acc:.4f} vs {ann_best_val_acc:.4f})")

if ann_best_val_loss < cnn_best_val_loss:
    print(f"ANN achieved lower validation loss ({ann_best_val_loss:.4f} vs {cnn_best_val_loss:.4f})")
else:
    print(f"CNN achieved lower validation loss ({cnn_best_val_loss:.4f} vs {ann_best_val_loss:.4f})")

# ============================================================================
# Section 6: Evaluation and Analysis
# ============================================================================

# Evaluate on test set
test_loss, test_accuracy = patternmind_model.evaluate(X_test, y_test_one_hot, verbose=0)

print("."*60)
print("TEST SET EVALUATION")
print("."*60)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss:     {test_loss:.4f}")
print("."*60)

# Get predictions
y_pred_probs = patternmind_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate comprehensive classification report
print("\n" + "."*60)
print("CLASSIFICATION REPORT")
print("."*60)
print("\nWhich categories does the model struggles to distinguish?")

# Get category names for the report
category_names = label_encoder.classes_

# Generate classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=category_names,
    digits=4
)

print(report)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(16, 14))

# Create heatmap
sns.heatmap(
    cm,
    annot=True,           # Show numbers in cells
    fmt='d',              # Integer format
    cmap='YlOrRd',        # Yellow-Orange-Red colormap
    xticklabels=category_names,
    yticklabels=category_names,
    cbar_kws={'label': 'Number of Predictions'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)

ax.set_title('Confusion Matrix: Revealing Visual Category Relationships',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Category', fontsize=13, fontweight='bold')
ax.set_ylabel('True Category', fontsize=13, fontweight='bold')

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=5)
plt.yticks(rotation=0, fontsize=5)

plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Find top misclassification pairs (excluding correct predictions on diagonal)
misclassifications = []

for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:  # Off-diagonal elements only
            misclassifications.append({
                'true_category': category_names[i],
                'predicted_category': category_names[j],
                'count': cm[i, j],
                'true_idx': i,
                'pred_idx': j
            })

# Sort by count
misclassifications.sort(key=lambda x: x['count'], reverse=True)

# Display top 20 misclassification pairs
print("."*80)
print("TOP 20 MISCLASSIFICATION PAIRS")
print("."*80)
print(f"{'Rank':<6} {'True Category':<20} {'-> Predicted As':<20} {'Count':<8} {'Pattern'}")
print("-"*80)

for rank, mc in enumerate(misclassifications[:20], 1):
    print(f"{rank:<6} {mc['true_category']:<20} → {mc['predicted_category']:<20} {mc['count']:<8}")

# Find indices of misclassified samples
misclassified_indices = np.where(y_test != y_pred)[0]

print(f"\n Total misclassified samples: {len(misclassified_indices)}")
print(f"Total test samples: {len(y_test)}")
print(f"Misclassification rate: {len(misclassified_indices)/len(y_test)*100:.2f}%\n")

# Get the prediction confidence (probability) for misclassified samples
misclassified_confidences = []
for idx in misclassified_indices:
    pred_prob = y_pred_probs[idx, y_pred[idx]]  # Probability of predicted class
    misclassified_confidences.append({
        'idx': idx,
        'true_label': y_test[idx],
        'pred_label': y_pred[idx],
        'confidence': pred_prob,
        'true_name': category_names[y_test[idx]],
        'pred_name': category_names[y_pred[idx]]
    })

# Sort by confidence (high confidence errors are most interesting)
misclassified_confidences.sort(key=lambda x: x['confidence'], reverse=True)

# Select top 16 most confident misclassifications for visualization
num_to_display = min(16, len(misclassified_confidences))
samples_to_show = misclassified_confidences[:num_to_display]

print(f"Displaying {num_to_display} most confident misclassifications...")
print("(High confidence errors reveal strong structural similarity)\n")

sns.set_style("whitegrid")
sns.set_palette("husl")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Confidence Distribution of Misclassifications
ax1 = fig.add_subplot(gs[0, :])
confidences = [s['confidence'] * 100 for s in misclassified_confidences]

sns.histplot(confidences, bins=30, kde=True, ax=ax1, color='coral', alpha=0.7)
ax1.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(confidences):.1f}%')
ax1.axvline(np.median(confidences), color='darkred', linestyle=':', linewidth=2,
            label=f'Median: {np.median(confidences):.1f}%')
ax1.set_xlabel('Prediction Confidence (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Confidence Scores for Misclassifications',
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# 2. Top 15 Most Misclassified Categories (True Labels)
ax2 = fig.add_subplot(gs[1, 0])

# Count misclassifications by true category
true_category_errors = {}
for sample in misclassified_confidences:
    true_cat = sample['true_name']
    true_category_errors[true_cat] = true_category_errors.get(true_cat, 0) + 1

# Sort and get top 15
true_sorted = sorted(true_category_errors.items(), key=lambda x: x[1], reverse=True)[:15]
true_df = pd.DataFrame(true_sorted, columns=['Category', 'Error Count'])

# Create horizontal bar plot
sns.barplot(data=true_df, y='Category', x='Error Count', ax=ax2,
            palette='Reds_r', orient='h')
ax2.set_xlabel('Number of Misclassifications', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Category', fontsize=11, fontweight='bold')
ax2.set_title('Top 15 Categories Most Often Misclassified\n(Hardest to Identify Correctly)',
              fontsize=12, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(true_df.iterrows()):
    ax2.text(row['Error Count'] + 0.5, i, str(row['Error Count']),
             va='center', fontsize=9, fontweight='bold')

# 3. Top 15 Most Over-Predicted Categories (Predicted Labels)
ax3 = fig.add_subplot(gs[1, 1])

# Count by predicted category
pred_category_errors = {}
for sample in misclassified_confidences:
    pred_cat = sample['pred_name']
    pred_category_errors[pred_cat] = pred_category_errors.get(pred_cat, 0) + 1

# Sort and get top 15
pred_sorted = sorted(pred_category_errors.items(), key=lambda x: x[1], reverse=True)[:15]
pred_df = pd.DataFrame(pred_sorted, columns=['Category', 'Count'])

# Create horizontal bar plot
sns.barplot(data=pred_df, y='Category', x='Count', ax=ax3,
            palette='Blues_r', orient='h')
ax3.set_xlabel('Number of Times Predicted', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Category', fontsize=11, fontweight='bold')
ax3.set_title('Top 15 Categories Most Often Over-Predicted\n(Model Tends to Over-Predict These)',
              fontsize=12, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(pred_df.iterrows()):
    ax3.text(row['Count'] + 0.5, i, str(row['Count']),
             va='center', fontsize=9, fontweight='bold')

# Sample misclassification
ax4 = fig.add_subplot(gs[2, :])

# Get top 20 samples with highest confidence
samples_to_show = sorted(misclassified_confidences,
                        key=lambda x: x['confidence'],
                        reverse=True)[:20]

# Prepare data
sample_nums = list(range(1, len(samples_to_show) + 1))
confidences_sample = [s['confidence'] * 100 for s in samples_to_show]
labels = [f"{s['true_name'][:20]}→{s['pred_name'][:20]}" for s in samples_to_show]

# Create bar plot
colors = sns.color_palette("RdYlGn_r", len(samples_to_show))
bars = ax4.barh(sample_nums, confidences_sample, color=colors, alpha=0.8, edgecolor='black')

ax4.set_yticks(sample_nums)
ax4.set_yticklabels([f"#{i}" for i in sample_nums], fontsize=9)
ax4.set_xlabel('Prediction Confidence (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Sample #', fontsize=11, fontweight='bold')
ax4.set_title('Top 20 High-Confidence Misclassifications',
              fontsize=12, fontweight='bold', pad=15)
ax4.grid(axis='x', alpha=0.3)
ax4.invert_yaxis()

# Add confidence values on bars
for i, (bar, conf) in enumerate(zip(bars, confidences_sample)):
    ax4.text(conf + 1, bar.get_y() + bar.get_height()/2,
             f'{conf:.1f}%', va='center', fontsize=8, fontweight='bold')

# Add overall statistics as text box
total_misclass = len(misclassified_confidences)
avg_conf = np.mean([s['confidence'] * 100 for s in misclassified_confidences])
median_conf = np.median([s['confidence'] * 100 for s in misclassified_confidences])

stats_text = f"""OVERALL STATISTICS
─────────────────────
Total Misclassifications: {total_misclass:,}
Average Confidence: {avg_conf:.1f}%
Median Confidence: {median_conf:.1f}%
Unique True Categories: {len(true_category_errors)}
Unique Predicted Categories: {len(pred_category_errors)}"""

fig.text(0.98, 0.02, stats_text, fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         verticalalignment='bottom', horizontalalignment='right')

plt.tight_layout()
plt.savefig('images/misclassification_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAnalysis complete!")
