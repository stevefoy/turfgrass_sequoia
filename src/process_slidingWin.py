#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:20:13 2024

@author: sfoy
"""
from PIL import Image
import os
import piexif
from tqdm import tqdm

def sliding_window_crop(input_dir, output_dir, window_size=(400, 400), step_size=100):
    """
    Resize images to a specified scale and apply sliding window cropping to all images in the input directory,
    then save them to the output directory.

    :param input_dir: Directory containing images to process.
    :param output_dir: Directory where cropped images will be saved.
    :param scale_factor: Decimal representing the scaling factor to resize images.
    :param window_size: The size of the cropping window (width, height).
    :param step_size: The number of pixels the window will move after each crop.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    new_size = None
    # Loop through all images in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            extension = os.path.splitext(filename)[1]
            exif_bytes = None
            
            # Not setup yet
            if filename.endswith("_RGB.JPG"):
                exif_dict = piexif.load(img.info["exif"])
                exif_bytes = piexif.dump(exif_dict)
            
                
            
            img_width, img_height = img.size

            # Generate crop coordinates and apply cropping
            for x in range(0, img_width - window_size[0] + 1, step_size):
                for y in range(0, img_height - window_size[1] + 1, step_size):
                    crop_box = (x, y, x + window_size[0], y + window_size[1])
                    cropped_img = img.crop(crop_box)
                    cropped_img_filename = f"{filename[:-4]}_scaled_1.0_crop_{x}_{y}"+extension
                    # cropped_img.save(os.path.join(output_dir, cropped_img_filename), exif=exif_bytes)
                    cropped_img.save(os.path.join(output_dir, cropped_img_filename))

            #print("Resizing and cropping completed. Size", img.size)

def resize_and_sliding_window_crop(input_dir, output_dir, scale_factor=0.5, window_size=(400, 400), step_size=100):
    """
    Resize images to a specified scale and apply sliding window cropping to all images in the input directory,
    then save them to the output directory.

    :param input_dir: Directory containing images to process.
    :param output_dir: Directory where cropped images will be saved.
    :param scale_factor: Decimal representing the scaling factor to resize images.
    :param window_size: The size of the cropping window (width, height).
    :param step_size: The number of pixels the window will move after each crop.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    new_size = None
    # Loop through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            
            # Resize the image based on the scale factor
            #new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            #img = img.resize(new_size, Image.ANTIALIAS)

            img_width, img_height = img.size

            # Generate crop coordinates and apply cropping
            for x in range(0, img_width - window_size[0] + 1, step_size):
                for y in range(0, img_height - window_size[1] + 1, step_size):
                    crop_box = (x, y, x + window_size[0], y + window_size[1])
                    cropped_img = img.crop(crop_box)
                    cropped_img_filename = f"{filename[:-4]}_scale_{scale_factor}_crop_{x}_{y}.png"
                    cropped_img.save(os.path.join(output_dir, cropped_img_filename))

    print("Resizing and cropping completed. Size", new_size )

def get_subdirectories(folder_path):
    try:
        subdirectories = [name for name in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, name))]
        return subdirectories
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return []
    

# List of folders
folders = [
    'ATU_01_MAY_2024', 'ATU_12_July_2023', 'ATU_24_APRIL_2024',
    'ATU_05_MAR_2024', 'ATU_14_MAY_2024', 'ATU_30_JAN_2024',
    'ATU_08_MAY_2024', 'ATU_19_FEB_2024', 'ATU_09_JUNE_2023',
    'ATU_20_MAR_2024'
]

base_folder_path = '/media/freddy/vault/datasets/Processed/'
base_folder_path = r'D:\datasets\Processed'

    
crop_folder = 'out_scale1.0_S224_v2'

# Directory walk to collect data and process images
for i, folder in enumerate(folders):
    # Example usage
    input_directory = os.path.join(base_folder_path, folder)
    #output_directory = os.path.join(input_directory,"out")
    subdirectories = get_subdirectories(input_directory)
    print("Subdirectories:", subdirectories)

    for input_subdirectory in subdirectories:
        input_dir_process = os.path.join(input_directory, input_subdirectory)
        print(input_dir_process)
        output_directory = os.path.join(input_dir_process,crop_folder)
        sliding_window_crop(input_dir_process, output_directory, window_size=(224, 224), step_size=112)

#for i in range(1,10,1):
#	i = round(0.1*i, 1)
#	sacler = "_scale"+str(i)
#	resize_and_sliding_window_crop(input_directory, output_directory+sacler, scale_factor=i, window_size=(400, 400), step_size=400)
