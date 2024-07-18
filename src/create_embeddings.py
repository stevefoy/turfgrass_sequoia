# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:32:59 2024

@author: stevf
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import logging

# Dataset class for loading images from a list of file paths
class ImageFileListDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.files = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Return image path as well

# Function to extract features using a pre-trained model
def extract_features(model, dataloader, embeddings_folder):
    # Ensure the embeddings folder exists
    os.makedirs(embeddings_folder, exist_ok=True)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    features = []
    paths = []  # To store image paths
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, batch_paths = data
            if torch.cuda.is_available():
                images = images.cuda()
            feature = model(images)
            features.append(feature.cpu().numpy())
            paths.extend(batch_paths)  # Append paths

            # Save the embeddings for each image
            for i in range(len(feature)):
                image_path = batch_paths[i]
                image_name = os.path.basename(image_path)
                embedding_path = os.path.join(embeddings_folder, image_name + '_embedding.npy')  # Define embedding path
                np.save(embedding_path, feature[i].cpu().numpy())  # Save the embedding

    return np.concatenate(features).reshape(len(dataloader.dataset), -1), paths

# Function to load embeddings from a folder
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

# Function to process image files and generate embeddings
def process_image_files(input_directory, base_image_name, crop_folder):
    base_name = os.path.splitext(base_image_name)[0]
    ndvi_rgb_file = base_name.replace("RGB", "NDVI_RGB") + ".png"
    nir_file = base_name.replace("RGB", "NIR.TIF")
    ndvi_file = base_name.replace("RGB", "NDVI.TIF")
    red_file = base_name.replace("RGB", "RED.TIF")

    files_to_check = [base_image_name]
    image_crops = []
    for file in files_to_check:
        image_path = os.path.join(input_directory, file)
        if os.path.isfile(image_path):
            image_crops.extend(check_cropped_images(input_directory, crop_folder, file))
        else:
            print(f"Image {image_path} does not exist.")

    return image_crops

# Function to check cropped images and process them
def check_cropped_images(input_directory, crop_folder, base_image_name):
    base_name = os.path.splitext(base_image_name)[0]
    scalesX = ["224", "336", "448", "560", "672", "784", "896"]
    offsetsY = ["0", "112", "224", "336", "448", "560", "672"]

    cropped_files = []
    for scale in scalesX:
        for offset in offsetsY:
            cropped_image_name = f"{base_name}_scaled_1.0_crop_{scale}_{offset}.png"
            cropped_image_path = os.path.join(input_directory, crop_folder, cropped_image_name)

            if os.path.isfile(cropped_image_path):
                cropped_files.append(cropped_image_path)

    return cropped_files

# Function to process a folder of images and generate embeddings
def process_folder(input_directory, image_list_file, crop_folder):
    all_rgb_crop = []
    with open(image_list_file, 'r') as file:
        images = [image.strip() for image in file.readlines()]
        for image in tqdm(images):
            all_rgb_crop.extend(process_image_files(input_directory, image, crop_folder))
    return all_rgb_crop

# Function to get image list files and process them
def get_image_list_files(main_folder, crop_folder):
    all_rgb_crop = []
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        image_list_file = os.path.join(subfolder_path, "imageList.txt")
        if os.path.isfile(image_list_file):
            all_rgb_crop.extend(process_folder(subfolder_path, image_list_file, crop_folder))
        else:
            print(f"File {image_list_file} does not exist.")

    return all_rgb_crop

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # List of folders
    folders = [
        'ATU_09_JUNE_2023', 'ATU_12_July_2023',
        'ATU_30_JAN_2024', 'ATU_19_FEB_2024', 'ATU_05_MAR_2024', 'ATU_20_MAR_2024',
        'ATU_24_APRIL_2024', 'ATU_01_MAY_2024', 'ATU_08_MAY_2024', 'ATU_14_MAY_2024', 'ATU_21_MAY_2024'
    ]

    #folders = ['ATU_21_MAY_2024']  # This seems like a temporary override for testing

    base_folder_path = r'D:\datasets\Processed'
    crop_folder = 'out_scale1.0_S224'

    # Directory walk to collect data and process images
    file_paths = []
    for folder in folders:
        input_directory = os.path.join(base_folder_path, folder)
        file_paths = get_image_list_files(input_directory, crop_folder)

        for i in file_paths:
            print(i)
        # Create the dataset and dataloader
        dataset = ImageFileListDataset(file_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        # Load pre-trained ResNet-50 model
        resnet_model = models.resnet50(pretrained=True)
        resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))  # Remove the classification layer

        # Load pre-trained EfficientNet-B7 model
        efficientnet_model = models.efficientnet_b7(pretrained=True)
        efficientnet_model = torch.nn.Sequential(*(list(efficientnet_model.children())[:-1]))  # Remove the classification layer

        # Extract features using ResNet-50
        embeddings_folder_resnet = os.path.join(input_directory, 'embeddings_Resnet')
        print("Extracting features using ResNet-50...")
        resnet_features, resnet_paths = extract_features(resnet_model, dataloader, embeddings_folder_resnet)

        # Extract features using EfficientNet-B7
        embeddings_folder_efficientnet = os.path.join(input_directory, 'embeddings_EfficientNet')
        print("Extracting features using EfficientNet-B7...")
        efficientnet_features, efficientnet_paths = extract_features(efficientnet_model, dataloader, embeddings_folder_efficientnet)
