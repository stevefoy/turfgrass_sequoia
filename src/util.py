#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: stevefoy
"""

import re
from datetime import datetime, timedelta
import os, glob
import exiftool

import platform
import tqdm
import json
import numpy as np

def save_homography_matrix(H, file_path):
    """Save the homography matrix to a JSON file."""
    H_list = H.tolist()  # Convert the matrix to a list
    with open(file_path, 'w') as f:
        json.dump(H_list, f)

def load_homography_matrix(file_path):
    """Load the homography matrix from a JSON file."""
    with open(file_path, 'r') as f:
        H_list = json.load(f)
    H = np.array(H_list)  # Convert the list back to a numpy array
    return H

def check_cropped_images(input_directory, crop_folder, base_image_path_name):
    base_image_name = os.path.basename(base_image_path_name)
    base_rgb_name = os.path.splitext(base_image_name)[0]
    ndvi_file = base_rgb_name.replace("RGB", "NDVI")
    scalesX = ["224", "336", "448", "560", "672", "784", "896"]
    offsetsY = ["0", "112", "224", "336", "448", "560", "672"]
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
                print("BAD", cropped_image_rgb_path, "  ", cropped_image_nir_path)
                raise Exception("STOP FAILING TO FIND FILES")
    return cropped_files


def process_image_files(input_directory, base_image_name, crop_folder):
    base_name = os.path.splitext(base_image_name)[0]
    files_to_check = [base_image_name]
    print("Cluster images ", files_to_check)
    image_crops = []
    image_path = os.path.join(input_directory, base_image_name)
    if os.path.isfile(image_path):
        image_crops.extend(check_cropped_images(input_directory, crop_folder, image_path))
    else:
        print(f"Image {image_path} does not exist.")
    return image_crops




def process_folder(input_directory, image_list_file, crop_folder):
    all_rgb_crop = []
    with open(image_list_file, 'r') as file:
        fulpath_images_list = [image.strip() for image in file.readlines()]
        for image in tqdm(fulpath_images_list ):
            all_rgb_crop.extend(process_image_files(input_directory, image, crop_folder))
    
    return all_rgb_crop


def get_image_list_files(main_folder, crop_folder):
    all_rgb_crop = []
    for subfolder in os.listdir(main_folder):
        if subfolder != "embeddings":
            subfolder_path = os.path.join(main_folder, subfolder)
            image_list_file = os.path.join(subfolder_path, "imageList.txt")
            if os.path.isfile(image_list_file):
                all_rgb_crop.extend(process_folder(subfolder_path, image_list_file, crop_folder))
            else:
                print(f"File {image_list_file} does not exist.")
    return all_rgb_crop



def get_exiftool_path():
    """Return the appropriate path to exiftool based on the operating system."""
    if platform.system() == "Windows":
        # Ensure the path to exiftool.exe is correct
        return "C:\\path\\to\\exiftool.exe"
    else:
        # Assume a typical Unix-like system path
        return "/usr/local/bin/exiftool"

def process_metadata(filepath):
    exiftool_path = get_exiftool_path()
    with exiftool.ExifToolHelper(exiftool_path) as exift:
        metaToolData = exift.get_metadata(filepath)[0]
    return metaToolData


# Regular expression to match the filenames
pattern = re.compile(r'IMG_(\d{6})_(\d{6})_(\d{4})_(\w+)\.(\w+)')

def find_related_files(filename, folder_path, typeImage="NIR"):
    match = pattern.match(filename)
    if not match:
        return []

    date, time, seq, band, ext = match.groups()
    time_obj = datetime.strptime(time, '%H%M%S')

    related_files = []
    # Loop over possible time variations (e.g., ±1 second)
    for delta in range(-1, 2):
        new_time_obj = time_obj + timedelta(seconds=delta)
        new_time_str = new_time_obj.strftime('%H%M%S')

        related_pattern = f'IMG_{date}_{new_time_str}_{seq}_{typeImage}.TIF'

        for file in os.listdir(folder_path):
            #print(file, "==", related_pattern)
            if related_pattern == file:
                related_files.append(file)
                
    return related_files