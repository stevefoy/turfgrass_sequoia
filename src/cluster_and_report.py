#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 03 10:52:08 2024

@author: stephen foy
"""

import os
from PIL import Image
import rasterio
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pickle
# Define functions
from sklearn.cluster import AgglomerativeClustering
# Disable logging
logging.basicConfig(level=logging.CRITICAL)
from scipy import stats


def clean_labels(labels):
    """Ensure labels are valid by replacing None with 'Unknown'."""
    return [label if label is not None else "Unknown" for label in labels]

def plot_tsne(embeddings, labels, title, folderName):
    """Plot t-SNE for the given embeddings and labels."""
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, max_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.array(labels) == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.savefig(folderName + "_" + title + ".png", dpi=300, bbox_inches='tight')
    
def plot_tsne_with_images(embeddings, image_paths, title, folderName, cache_dir='cache'):
    """Plot t-SNE for the given embeddings and overlay the corresponding images."""
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Define cache file path
    cache_file = os.path.join(cache_dir, f'{folderName}_tsne_results.pkl')
    
    if os.path.exists(cache_file):
        # Load t-SNE results from cache
        with open(cache_file, 'rb') as f:
            tsne_results = pickle.load(f)
        print("Loaded t-SNE results from cache.")
    else:
        # Compute t-SNE results
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)
        
        # Save t-SNE results to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(tsne_results, f)
        print("Computed and saved t-SNE results to cache.")
    
    fig, ax = plt.subplots(figsize=(30, 30))

    artists = []
    for xy, path in zip(tsne_results, image_paths):
        x0, y0 = xy
        img = Image.open(path)
        img = img.resize((10, 10), Image.LANCZOS)  # Resize the image thumbnail
        im = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        
    ax.update_datalim(tsne_results)
    ax.autoscale()
    ax.set_title(title)
    plt.savefig(folderName + ".png", dpi=300, bbox_inches='tight')
    
def plot_tsne_with_imagesV1(embeddings, image_paths, title, folderName):
    """Plot t-SNE for the given embeddings and overlay the corresponding images."""
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(20, 20))

    artists = []
    for xy, path in zip(tsne_results, image_paths):
        x0, y0 = xy
        img = Image.open(path)
        img = img.resize((26, 26), Image.LANCZOS)  # Resize the image thumbnail
        im = OffsetImage(img, zoom=1)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        
    ax.update_datalim(tsne_results)
    ax.autoscale()
    ax.set_title(title)
    plt.savefig(folderName + ".png", dpi=300, bbox_inches='tight')

def read_normalized_ndvi_image(ndvi_image_path):
    """Read and normalize NDVI image get mean"""
    try:
        with rasterio.open(ndvi_image_path) as src:
            ndvi = src.read(1)
            ndvi_normalized = (ndvi + 1) / 2  # Normalize NDVI to range 0-1
            ndvi_normalized = np.clip(ndvi_normalized, 0, 1.0)
            
            border_value = 340282346638528859811704183484516925440.00
            # Create a mask for the border values
            mask = ndvi <= 1.1
            
            # Apply the mask to the normalized NDVI image
            ndvi_masked = np.where(mask, ndvi_normalized, np.nan)
            
            # Calculate the mean value excluding the borders
            mean_value = np.nanmean(ndvi_masked)
            #mode_value = stats.mode(ndvi_masked, nan_policy='omit').mode[0]
            
            # Print the highest and smallest NDVI values excluding the masked values
            highest_value = np.nanmax(ndvi_masked)
            smallest_value = np.nanmin(ndvi_masked)
            print(f"Highest NDVI value (excluding masked): {highest_value}")
            print(f"Smallest NDVI value (excluding masked): {smallest_value}")

            
            return ndvi_masked, mean_value
        
    except Exception as e:
        print("Exceptiom ", e)
        
        return None, None



def classify_damage(mean_ndvi):
    """Classify damage based on the mean NDVI value."""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i, threshold in enumerate(thresholds):
        if mean_ndvi < threshold:
            return f"NDVI{i+1}"
    return "NDVI10"

def assess_damage(ndvi):
    """Assess damage levels for given NDVI data."""
    damage_levels = {
        "NDVI1": (0.0, 0.1),
        "NDVI2": (0.1, 0.2),
        "NDVI3": (0.2, 0.3),
        "NDVI4": (0.3, 0.4),
        "NDVI5": (0.4, 0.5),
        "NDVI6": (0.5, 0.6),
        "NDVI7": (0.6, 0.7),
        "NDVI8": (0.7, 0.8),
        "NDVI9": (0.8, 0.9),
        "NDVI10": (0.9, 1.0)
    }
    
    damage_assessment = {}
    valid_pixels = np.isfinite(ndvi)  # Only consider valid (non-NaN) pixels
    total_valid_pixels = np.sum(valid_pixels)
    
    for level, (min_val, max_val) in damage_levels.items():
        mask = (ndvi >= min_val) & (ndvi < max_val) & valid_pixels
        damage_assessment[level] = np.sum(mask) / total_valid_pixels * 100
        
    return damage_assessment

def load_embeddings_from_file_paths(file_paths, embeddings_folder):
    embeddings = []
    paths = []
    
    Resnet = False
    Efficient = False
    if "Efficient" in embeddings_folder:
        Efficient = True
    elif "Resnet" in embeddings_folder:
        Resnet = True
    
    for (file_rgb_path, file_nir_path) in file_paths:
        base_name = os.path.basename(file_rgb_path)
        embedding_file = f"{base_name}_embedding.npy"
        embedding_path = os.path.join(embeddings_folder, embedding_file)
        try:
            embedding = np.load(embedding_path)
            if Resnet:
                embedding = embedding.reshape(2048)                
            elif Efficient:
                embedding = embedding.reshape(2560)
                
            embeddings.append(embedding)
            paths.append(file_rgb_path)
           
        except Exception as e:
            logging.error(f"Error loading embedding {embedding_path}: {e}")
    
    embeddings = np.array(embeddings)
    return embeddings, paths

def get_image_list_files(main_folder, crop_folder):
    all_rgb_crop = []
    for subfolder in os.listdir(main_folder):
        if subfolder != "embeddings" or subfolder != "embeddings_Resnet":
            subfolder_path = os.path.join(main_folder, subfolder)
            image_list_file = os.path.join(subfolder_path, "imageList.txt")
            if os.path.isfile(image_list_file):
                all_rgb_crop.extend(process_folder(subfolder_path, image_list_file, crop_folder))
            else:
                print(f"File {image_list_file} does not exist.")
            
    return all_rgb_crop

def process_image_files(input_directory, base_image_name, crop_folder):
    base_name = os.path.splitext(base_image_name)[0]
    files_to_check = [base_image_name]
    image_crops = []
    image_path = os.path.join(input_directory, base_image_name)
    if os.path.isfile(image_path):
        image_crops.extend(check_cropped_images(input_directory, crop_folder, image_path))
    else:
        print(f"Image {image_path} does not exist.")
    return image_crops

def check_cropped_images(input_directory, crop_folder, base_image_path_name):
    base_image_name = os.path.basename(base_image_path_name)
    base_rgb_name = os.path.splitext(base_image_name)[0]
    ndvi_file = base_rgb_name.replace("RGB", "NDVI_ALIGNED")
    scalesX = ["224", "448", "672", "896"]
    offsetsY = ["224", "448", "672"]
    cropped_files = []
    for scale in scalesX:
        for offset in offsetsY:
            cropped_image_rgb_name = f"{base_rgb_name}_scaled_1.0_crop_{scale}_{offset}.png"
            cropped_image_rgb_path = os.path.join(input_directory, crop_folder, cropped_image_rgb_name)
            cropped_image_nir_name = f"{ndvi_file}_scaled_1.0_crop_{scale}_{offset}.TIF"
            cropped_image_nir_path = os.path.join(input_directory, crop_folder, cropped_image_nir_name)
            if os.path.isfile(cropped_image_rgb_path) and os.path.isfile(cropped_image_nir_path):
                cropped_files.append([cropped_image_rgb_path, cropped_image_nir_path])
            else:
                print(cropped_image_nir_path," ",  cropped_image_rgb_path)
                raise Exception("STOP FAILING TO FIND FILES")
    return cropped_files

def process_folder(input_directory, image_list_file, crop_folder):
    all_rgb_crop = []
    with open(image_list_file, 'r') as file:
        images = [image.strip() for image in file.readlines()]
        for image in tqdm(images):
            all_rgb_crop.extend(process_image_files(input_directory, image, crop_folder))
    return all_rgb_crop

def cluster_embeddings(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def group_filenames_by_cluster(paths, cluster_labels):
    cluster_dict = {}
    for path, label in zip(paths, cluster_labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(path)
    return cluster_dict

def check_file_suffix(filepath):
    offsetFile = ["_0.TIF", "_112.TIF", "_224.TIF", "_336.TIF", "_448.TIF", "_560.TIF", "_672.TIF"]
    return any(filepath.endswith(suffix) for suffix in offsetFile)

def compare_clusters_to_ground_truth(cluster_dict, damage_levels):
    cluster_comparisons = {}

    for cluster, file_paths in tqdm(cluster_dict.items()):
        cluster_comparisons[cluster] = {level: 0 for level in damage_levels.keys()}
        
        for file_path in tqdm(file_paths):
            file_path_NDVI = file_path.replace("RGB", "NDVI_ALIGNED").replace(".png", ".TIF")
            ndvi_image_path = file_path_NDVI
            if os.path.isfile(ndvi_image_path):

                ndvi, mean_ndvi = read_normalized_ndvi_image(ndvi_image_path)
                
                if ndvi is not None:
                    damage_assessment = assess_damage(ndvi)
                    for level in damage_assessment:
                        cluster_comparisons[cluster][level] += damage_assessment[level]
                    # mean_ndvi = np.mean(ndvi)
                    damage_level = classify_damage(mean_ndvi)
                    if 0:
                        debug_view_image(file_path, ndvi)
                        debug_view_image_raw(file_path, ndvi_image_path)
                        print("Damage Assessment:")
                        for level, percentage in damage_assessment.items():
                            print(f"{level}: {percentage:.2f}%")
        
        num_files = len(file_paths)
        
        for level in cluster_comparisons[cluster]:
            cluster_comparisons[cluster][level] /= num_files
    
    return cluster_comparisons

def get_ground_truth_labels(file_paths, damage_levels):
    ground_truth_labels = []
    for file_path in file_paths:
        ndvi_path = file_path.replace("RGB", "NDVI_ALIGNED").replace(".png", ".TIF")
        logging.info(f"Processing NDVI path: {ndvi_path}")
        #mean_ndvi = read_normalized_ndvi_image(ndvi_path)
        ndvi, mean_ndvi = read_normalized_ndvi_image(ndvi_path)
        if mean_ndvi is not None:
            #mean_ndvi = np.mean(ndvi)
            label = classify_damage(mean_ndvi)
            logging.info(f"Mean NDVI: {mean_ndvi}, Label: {label}")
            ground_truth_labels.append(label)
        else:
            logging.error(f"Failed to process NDVI path: {ndvi_path}")
    return ground_truth_labels

def assign_cluster_labels(cluster_dict, damage_levels):
    cluster_labels = {}
    for cluster, file_paths in cluster_dict.items():
        ground_truth_labels = get_ground_truth_labels(file_paths, damage_levels)
        if ground_truth_labels:
            most_common_label = max(set(ground_truth_labels), key=ground_truth_labels.count)
            cluster_labels[cluster] = most_common_label
        else:
            logging.error(f"No ground truth labels found for cluster {cluster}")
    return cluster_labels

def map_cluster_labels_to_files(cluster_dict, cluster_labels):
    file_label_mapping = []
    for cluster, file_paths in cluster_dict.items():
        label = cluster_labels[cluster]
        for file_path in file_paths:
            file_label_mapping.append((file_path, label))
    return file_label_mapping

def get_true_labels(file_label_mapping, damage_levels):
    true_labels = []
    for file_path, _ in file_label_mapping:
        ndvi_path = file_path.replace('RGB', 'NDVI_ALIGNED').replace('.png', '.TIF')
        logging.info(f"Processing NDVI path: {ndvi_path}")
        
        ndvi, mean_ndvi = read_normalized_ndvi_image(ndvi_path)
        if mean_ndvi is not None:
            
            true_label = classify_damage(mean_ndvi)
            logging.info(f"Mean NDVI: {mean_ndvi}, Label: {true_label}")
            true_labels.append(true_label)
        else:
            logging.error(f"Failed to process NDVI path: {ndvi_path}")
    return true_labels

def get_contingency_matrix(true_labels, predicted_labels, classes):
    contingency_matrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
    return contingency_matrix

def match_labels_with_hungarian_algorithm(contingency_matrix):
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return label_mapping

def map_labels(predicted_labels, label_mapping):
    mapped_labels = [label_mapping[label] if label in label_mapping else label for label in predicted_labels]
    return mapped_labels

# Main script

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    folders = [
        'ATU_09_JUNE_2023',
        'ATU_30_JAN_2024', 'ATU_19_FEB_2024', 'ATU_05_MAR_2024', 'ATU_20_MAR_2024',
        'ATU_24_APRIL_2024','ATU_01_MAY_2024','ATU_08_MAY_2024', 'ATU_14_MAY_2024'
    ]
    # folders = ['ATU_09_JUNE_2023'], 
    base_folder_path = r'C:\Users\stevf\OneDrive\Documents\Projects\Github_IMVIP2024\Processed'
    crop_folder = 'out_scale1.0_S224_v2' 

    embeddings_folder_name = 'embeddings'
 #embeddings_EfficientNet
 # embeddings_Resnet
 
    damage_levels = {
        "NDVI1": (0.0, 0.1),
        "NDVI2": (0.1, 0.2),
        "NDVI3": (0.2, 0.3),
        "NDVI4": (0.3, 0.4),
        "NDVI5": (0.4, 0.5),
        "NDVI6": (0.5, 0.6),
        "NDVI7": (0.6, 0.7),
        "NDVI8": (0.7, 0.8),
        "NDVI9": (0.8, 0.9),
        "NDVI10": (0.9, 1.0)
    }

    all_true_labels = []
    all_predicted_labels = []
    all_embeddings = []
    all_paths = []

    for folder in folders:
        input_directory = os.path.join(base_folder_path, folder)
        file_paths = get_image_list_files(input_directory, crop_folder)

        embeddings_folder = os.path.join(input_directory, embeddings_folder_name)
        embeddings, paths = load_embeddings_from_file_paths(file_paths, embeddings_folder)

        all_embeddings.append(embeddings)
        all_paths.extend(paths)
    
	# Debugging: Check shapes of all_embeddings before concatenation
    for i, embedding in enumerate(all_embeddings):
        print(f"Embedding {i} shape: {embedding.shape}")
        
    # Concatenate all embeddings from all folders
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalize(all_embeddings))
    
    """
    silhouette_scores = 0
    em = normalize(all_embeddings, norm='l2')
    for k in range(2,10,1 ):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(em)
        silhouette_scores = silhouette_score(em, cluster_labels)
        print("SCORE", k , "  R ", silhouette_scores)
   
    silhouette_scores = 0
    em = normalize(all_embeddings)
    for k in range(2,10,1 ):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(em)
        silhouette_scores = silhouette_score(em, cluster_labels)
        print("SCORE", k , "  R ", silhouette_scores)
      
    em = normalize(all_embeddings, norm='l2')
    agglo = AgglomerativeClustering(n_clusters=5, metric='cosine', linkage='average')
    cluster_labels = agglo.fit_predict(em)
"""
    cluster_dict = group_filenames_by_cluster(all_paths, cluster_labels)
    cluster_comparisons = compare_clusters_to_ground_truth(cluster_dict, damage_levels)

    cluster_labels_assigned = assign_cluster_labels(cluster_dict, damage_levels)
    file_label_mapping = map_cluster_labels_to_files(cluster_dict, cluster_labels_assigned)
    true_labels = get_true_labels(file_label_mapping, damage_levels)
    predicted_labels = [label for _, label in file_label_mapping]

    all_true_labels = clean_labels(true_labels)
    all_predicted_labels = clean_labels(predicted_labels)

    # Filter classes to include only those present in true_labels
    classes = [cls for cls in damage_levels.keys() if cls in all_true_labels]

    if not classes:
        logging.error("No valid classes found in true_labels. Skipping further analysis.")
    else:
        logging.info(f"True labels: {all_true_labels}")
        logging.info(f"Predicted labels: {all_predicted_labels}")
        logging.info(f"Classes: {classes}")

        contingency_matrix = get_contingency_matrix(all_true_labels, all_predicted_labels, classes)

        label_mapping = match_labels_with_hungarian_algorithm(contingency_matrix)
        mapped_labels = map_labels(all_predicted_labels, label_mapping)

        conf_matrix = confusion_matrix(all_true_labels, mapped_labels, labels=classes)
        class_report = classification_report(all_true_labels, mapped_labels, labels=classes, target_names=classes)
        f1 = f1_score(all_true_labels, mapped_labels, average='weighted')

        # Save results to file
        output_file = "results_combined.txt"
        with open(output_file, 'a') as f:
            f.write("Combined Results:\n")
            f.write("Contingency Matrix:\n")
            f.write(np.array2string(contingency_matrix) + "\n\n")
        
            f.write("Label Mapping:\n")
            f.write(str(label_mapping) + "\n\n")
        
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(conf_matrix) + "\n\n")
        
            f.write("Classification Report:\n")
            f.write(class_report + "\n\n")
        
            f.write(f"Weighted F1 Score: {f1:.4f}\n\n")
        
            for cluster, comparisons in cluster_comparisons.items():
                f.write(f"Cluster {cluster} comparisons:\n")
                for level, percentage in comparisons.items():
                    f.write(f"{level}: {percentage:.2f}%\n")
                f.write("\n")

        for cluster, comparisons in cluster_comparisons.items():
            logging.info(f"Cluster {cluster} comparisons: {comparisons}")

        # Ensure embeddings is a NumPy array
        if isinstance(all_embeddings, list):
            all_embeddings = np.vstack(all_embeddings)

        plot_tsne(all_embeddings, all_predicted_labels, "t-SNE with Cluster Labels", "Combined")
        plot_tsne(all_embeddings, all_true_labels, "t-SNE with True Labels", "Combined")
        plot_tsne_with_images(all_embeddings, all_paths, "t-SNE with Images", "Combined")
