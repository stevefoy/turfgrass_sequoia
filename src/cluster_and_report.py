#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:52:08 2024

@author: stephen
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
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

def debug_view_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file found at {image_path}")

    with Image.open(image_path) as img:
        ndvi_data = np.array(img)

    data_min = ndvi_data.min()
    data_max = ndvi_data.max()
    print(f"Data Min: {data_min}, Data Max: {data_max}")

    ndvi_min = 0.0
    ndvi_max = 1.0

    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi_data, cmap='RdYlGn', vmin=ndvi_min, vmax=ndvi_max)
    plt.colorbar(label='NDVI')
    plt.title("NDVI Image")
    plt.axis('off')
    plt.show()

def debug_view_image_NDIV(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file found at {image_path}")

    with Image.open(image_path) as img:
        ndvi_data = np.array(img)

    data_min = ndvi_data.min()
    data_max = ndvi_data.max()
    print(f"Data Min: {data_min}, Data Max: {data_max}")

    ndvi_min = -1.0
    ndvi_max = 1.0

    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi_data, cmap='RdYlGn', vmin=ndvi_min, vmax=ndvi_max)
    plt.colorbar(label='NDVI')
    plt.title("NDVI Image")
    plt.axis('off')
    plt.show()

def read_normalized_ndvi_image(ndvi_image_path):
    with rasterio.open(ndvi_image_path) as src:
        ndvi = src.read(1)
        #data_min = ndvi.min()
        #data_max = ndvi.max()

        
        # Normalize using the full range of NDVI values (-1 to +1)
        ndvi_normalized = (ndvi + 1) / 2  # Maps -1 to 0 and +1 to 1
        
        # Clip the values to the range [0, 1]
        ndvi_normalized = np.clip(ndvi_normalized, 0, 1.0)
    
    return ndvi_normalized

def read_normalized_ndvi_image_topremove(ndvi_image_path, ndvi_min=0, ndvi_max=1.0, offset=24):
    with rasterio.open(ndvi_image_path) as src:
        width = src.width
        height = src.height
        ndvi = src.read(1, window=((offset, height), (0, width))).astype(np.float32)
        # Normalize using the full range of NDVI values (-1 to +1)
        ndvi_normalized = (ndvi + 1) / 2  # Maps -1 to 0 and +1 to 1
        
        # Clip the values to the range [0, 1]
        ndvi_normalized = np.clip(ndvi_normalized, 0, 1.0)
    
    return ndvi_normalized

def classify_damage(mean_ndvi):
    if mean_ndvi < 0.2:
        return "No Damage"
    elif 0.2 <= mean_ndvi < 0.5:
        return "Low Damage"
    elif 0.5 <= mean_ndvi < 0.7:
        return "Moderate Damage"
    elif 0.7 >= mean_ndvi:
        return "High Damage"

def assess_damage(ndvi):
    damage_levels = {
        "No Damage": (0, 0.2),
        "Low Damage": (0.2, 0.5),
        "Moderate Damage": (0.5, 0.7),
        "High Damage": (0.7, 1.0)
    }

    damage_assessment = {}
    total_pixels = ndvi.size
    for level, (min_val, max_val) in damage_levels.items():
        mask = (ndvi >= min_val) & (ndvi < max_val)
        damage_assessment[level] = np.sum(mask) / total_pixels * 100
    
    return damage_assessment

def load_embeddings_from_folder(crops_folder, embeddings_folder):
    embeddings = []
    paths = []
    for file in os.listdir(embeddings_folder):
        if file.endswith('_embedding.npy'):
            embedding_path = os.path.join(embeddings_folder, file)
            image_path = os.path.join(crops_folder, file.replace('_embedding.npy', ''))
            try:
                embedding = np.load(embedding_path)
                embeddings.append(embedding)
                paths.append(image_path)
            except Exception as e:
                logging.error(f"Error loading embedding {embedding_path}: {e}")
    return np.array(embeddings), paths

def load_embeddings_from_file_paths(file_paths, embeddings_folder):
    embeddings = []
    paths = []
    for (file_rgb_path, file_nir_path) in file_paths:
        base_name = os.path.basename(file_rgb_path)
        embedding_file = f"{base_name}_embedding.npy"
        embedding_path = os.path.join(embeddings_folder, embedding_file)
        try:
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            paths.append([file_rgb_path, file_nir_path])
        except Exception as e:
            logging.error(f"Error loading embedding {embedding_path}: {e}")
    return np.array(embeddings), paths









def cluster_embeddings(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def group_filenames_by_cluster(paths, cluster_labels):
    cluster_dict = {}
    for path_array, label in zip(paths, cluster_labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(path_array)
    return cluster_dict

def check_file_suffix(filepath):
    offsetFile = ["_0.TIF", "_112.TIF", "_224.TIF", "_336.TIF", "_448.TIF", "_560.TIF", "_672.TIF"]
    return any(filepath.endswith(suffix) for suffix in offsetFile)

def compare_clusters_to_ground_truth(cluster_dict, damage_levels):
    cluster_comparisons = {}
    for cluster, file_paths in tqdm(cluster_dict.items()):
        cluster_comparisons[cluster] = {
            "No Damage": 0,
            "Low Damage": 0,
            "Moderate Damage": 0,
            "High Damage": 0
        }
        for file_path, file_path_NDVI in tqdm(file_paths):
            ndvi_image_path = file_path_NDVI
            if os.path.isfile(ndvi_image_path):
                if check_file_suffix(ndvi_image_path):
                    ndvi = read_normalized_ndvi_image_topremove(ndvi_image_path)
                else:
                    ndvi = read_normalized_ndvi_image(ndvi_image_path)
                mean_ndvi = np.mean(ndvi)
                damage_level = classify_damage(mean_ndvi)
                damage_assessment = assess_damage(ndvi)
                cluster_comparisons[cluster] = assess_damage(ndvi)
                if damage_level == "High Damage":
                    debug_view_image(ndvi_image_path)
                    print("Damage Assessment:")
                    for level, percentage in damage_assessment.items():
                        print(f"{level}: {percentage:.2f}%")
        num_files = len(file_paths)
        for level in cluster_comparisons[cluster]:
            cluster_comparisons[cluster][level] /= num_files
    return cluster_comparisons

def get_ground_truth_labels(file_paths, damage_levels):
    ground_truth_labels = []
    for _, ndvi_path in file_paths:
        ndvi = read_normalized_ndvi_image(ndvi_path)
        mean_ndvi = np.mean(ndvi)
        label = classify_damage(mean_ndvi)
        ground_truth_labels.append(label)
    return ground_truth_labels

def assign_cluster_labels(cluster_dict, damage_levels):
    cluster_labels = {}
    for cluster, file_paths in cluster_dict.items():
        ground_truth_labels = get_ground_truth_labels(file_paths, damage_levels)
        most_common_label = max(set(ground_truth_labels), key=ground_truth_labels.count)
        cluster_labels[cluster] = most_common_label
    return cluster_labels

def map_cluster_labels_to_files(cluster_dict, cluster_labels):
    file_label_mapping = []
    for cluster, file_paths in cluster_dict.items():
        label = cluster_labels[cluster]
        for file_path in file_paths:
            file_label_mapping.append((file_path[0], label))
    return file_label_mapping

def get_true_labels(file_label_mapping, damage_levels):
    true_labels = []
    for file_path, _ in file_label_mapping:
        ndvi_path = file_path.replace('RGB', 'NDVI').replace('.png', '.TIF')
        ndvi = read_normalized_ndvi_image(ndvi_path)
        mean_ndvi = np.mean(ndvi)
        true_label = classify_damage(mean_ndvi)
        true_labels.append(true_label)
    return true_labels

def get_contingency_matrix(true_labels, predicted_labels, classes):
    """
    Generate the contingency matrix for true and predicted labels.
    
    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.
        classes (list): List of class names.
    
    Returns:
        np.ndarray: Contingency matrix.
    """
    contingency_matrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
    return contingency_matrix

def match_labels_with_hungarian_algorithm(contingency_matrix):
    """
    Use the Hungarian algorithm to match predicted cluster labels with ground truth labels.
    
    Args:
        contingency_matrix (np.ndarray): Contingency matrix.
    
    Returns:
        dict: Mapping from predicted cluster labels to ground truth labels.
    """
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return label_mapping

def map_labels(predicted_labels, label_mapping):
    """
    Map predicted labels to the ground truth labels based on the label mapping.
    
    Args:
        predicted_labels (list): List of predicted labels.
        label_mapping (dict): Mapping from predicted cluster labels to ground truth labels.
    
    Returns:
        list: List of mapped labels.
    """
    mapped_labels = [label_mapping[label] for label in predicted_labels]
    return mapped_labels

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    folders = [
        'ATU_12_July_2023', 'ATU_24_APRIL_2024',
        'ATU_05_MAR_2024', 'ATU_14_MAY_2024', 'ATU_30_JAN_2024',
        'ATU_08_MAY_2024', 'ATU_19_FEB_2024', 'ATU_09_JUNE_2023',
        'ATU_20_MAR_2024'
    ]
    folders = ['ATU_05_MAR_2024']

    base_folder_path = '/media/freddy/vault/datasets/Processed/'
    crop_folder = 'out_scale1.0_S224'
    embeddings_folder_name = 'embeddings'

    damage_levels = {
        "No Damage": (0, 0.2),
        "Low Damage": (0.2, 0.5),
        "Moderate Damage": (0.5, 0.7),
        "High Damage": (0.7, 1.0)
    }
    
    all_true_labels = []
    all_predicted_labels = []
    
    for folder in folders:
        input_directory = os.path.join(base_folder_path, folder)
        file_paths = get_image_list_files(input_directory, crop_folder)

        embeddings_folder = os.path.join(input_directory, embeddings_folder_name)
        embeddings, paths = load_embeddings_from_file_paths(file_paths, embeddings_folder)
        print("Loaded embeddings:", embeddings)
        print("Corresponding paths:", paths)
        
        n_clusters = 4
        cosine_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = cosine_kmeans.fit_predict(normalize(embeddings))
        print(f"Cluster labels: {cluster_labels}")
        
        cluster_dict = group_filenames_by_cluster(paths, cluster_labels)
        cluster_comparisons = compare_clusters_to_ground_truth(cluster_dict, damage_levels)

        cluster_labels_assigned = assign_cluster_labels(cluster_dict, damage_levels)
        file_label_mapping = map_cluster_labels_to_files(cluster_dict, cluster_labels_assigned)
        true_labels = get_true_labels(file_label_mapping, damage_levels)
        predicted_labels = [label for _, label in file_label_mapping]

        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)

    classes = list(damage_levels.keys())
    contingency_matrix = get_contingency_matrix(all_true_labels, all_predicted_labels, classes)
    print("Contingency Matrix:")
    print(contingency_matrix)

    label_mapping = match_labels_with_hungarian_algorithm(contingency_matrix)
    print("Label Mapping:", label_mapping)

    # Handling KeyError: map only if label exists in label_mapping
    mapped_labels = [label_mapping[label] if label in label_mapping else label for label in all_predicted_labels]

    conf_matrix = confusion_matrix(all_true_labels, mapped_labels, labels=classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    class_report = classification_report(all_true_labels, mapped_labels, labels=classes, target_names=classes)
    print("Classification Report:")
    print(class_report)

    f1 = f1_score(all_true_labels, mapped_labels, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")

    for cluster, comparisons in cluster_comparisons.items():
        print(f"Cluster {cluster} comparisons: {comparisons}")
