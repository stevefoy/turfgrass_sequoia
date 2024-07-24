#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:20:13 2024

@author: sfoy
"""

from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def mask_outside_roi(image_np, roi_x1, roi_y1, roi_x2, roi_y2, max_value):
    """
    Mask the area outside the region of interest (ROI) with the maximum value.
    """
    image_np[:roi_y1, :] = max_value  # Top
    image_np[roi_y2:, :] = max_value  # Bottom
    image_np[:, :roi_x1] = max_value  # Left
    image_np[:, roi_x2:] = max_value  # Right
    return image_np

def process_image(filename, input_dir, output_dir, window_size=(400, 400), step_size=100):
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)
    extension = os.path.splitext(filename)[1]
    
    img_width, img_height = img.size
    img_np = np.array(img)

    if filename.endswith('NDVI_ALIGNED.TIF'):
        if img_np.dtype.kind in 'iu':  # Integer types
            max_value = np.iinfo(img_np.dtype).max
        elif img_np.dtype.kind == 'f':  # Floating point types
            max_value = np.finfo(img_np.dtype).max
        
        # Mask the area outside the ROI with the maximum value
        img_np = mask_outside_roi(img_np, 18, 16, 1256, 916, max_value)
        img = Image.fromarray(img_np)
    
    # Generate crop coordinates and apply cropping
    for x in range(0, img_width - window_size[0] + 1, step_size):
        for y in range(0, img_height - window_size[1] + 1, step_size):
            crop_box = (x, y, x + window_size[0], y + window_size[1])
            cropped_img = img.crop(crop_box)
            cropped_img_filename = f"{filename[:-4]}_scaled_1.0_crop_{x}_{y}{extension}"
                                 
            cropped_img.save(os.path.join(output_dir, cropped_img_filename))

def sliding_window_crop(input_dir, output_dir, window_size=(400, 400), step_size=100):
    """
    Resize images to a specified scale and apply sliding window cropping to all images in the input directory,
    then save them to the output directory.

    :param input_dir: Directory containing images to process.
    :param output_dir: Directory where cropped images will be saved.
    :param window_size: The size of the cropping window (width, height).
    :param step_size: The number of pixels the window will move after each crop.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image filenames
    image_filenames = [f for f in os.listdir(input_dir) if (f.endswith(('_RGB.png', 'NDVI_ALIGNED.TIF')) and not f.endswith('NDVI_RGB.png'))]

    # Process images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, filename, input_dir, output_dir, window_size, step_size) for filename in image_filenames]
        for future in tqdm(futures):
            future.result()

def get_subdirectories(folder_path):
    try:
        subdirectories = [name for name in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, name))]
        return subdirectories
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return []
    

# List of folders
folders = {
    'ATU_01_MAY_2024':   ['0007', '0008', '0009', '0010', '0011'],
    'ATU_12_July_2023':  ['0110', '0111', '0113'],
    'ATU_24_APRIL_2024': ['0005'],
    'ATU_05_MAR_2024':   ['0128'],
    'ATU_14_MAY_2024':   ['0006'],
    'ATU_30_JAN_2024':   ['0118'],
    'ATU_08_MAY_2024':   ['0007'],
    'ATU_19_FEB_2024':   ['0123'],
    'ATU_09_JUNE_2023':  ['0094', '0095'],
    'ATU_20_MAR_2024':   ['0001', '0002'],
    'ATU_21_MAY_2024':   ['0016'],
}

base_folder_path = r'C:\Users\stevf\OneDrive\Documents\Projects\Github_IMVIP2024\Processed'
crop_folder = 'out_scale1.0_S224_v2'

# Directory walk to collect data and process images
for folder, specific_subdirs in folders.items():
    input_directory = os.path.join(base_folder_path, folder)
    subdirectories = get_subdirectories(input_directory)
    
    if specific_subdirs:
        subdirectories = [subdir for subdir in subdirectories if subdir in specific_subdirs]
    
    print(f"Processing folder: {folder}")
    print("Subdirectories:", subdirectories)

    for input_subdirectory in subdirectories:
        input_dir_process = os.path.join(input_directory, input_subdirectory)
        print(f"Processing subdirectory: {input_dir_process}")
        output_directory = os.path.join(input_dir_process, crop_folder)
        sliding_window_crop(input_dir_process, output_directory, window_size=(224, 224), step_size=224)

# Uncomment to perform resizing and sliding window cropping with different scale factors
# for i in range(1, 10, 1):
#     scale_factor = round(0.1 * i, 1)
#     scaler_folder = f"_scale{scale_factor}"
#     resize_and_sliding_window_crop(input_directory, output_directory + scaler_folder, scale_factor=scale_factor, window_size=(400, 400), step_size=400)
