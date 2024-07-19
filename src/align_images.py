#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2. Align the RED LENS to the NIR Channel save, additionally calculate NDVI value

@author: stevefoy
"""


import cv2
import numpy as np
import os
import argparse
import config
import util
import util_sequoi
import matplotlib.colors as mcolors
from tqdm import tqdm
from PIL import Image

import tifffile as tiff
import rasterio
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

colors = list(mcolors.TABLEAU_COLORS.values())

import cv2
import numpy as np

def translate_image(image, tx, ty):
    """
    Translates the image using the specified offsets.
    
    Args:
    image (numpy.ndarray): The image to be translated.
    tx (int): The offset in pixels to translate along the x-axis.
    ty (int): The offset in pixels to translate along the y-axis.
    
    Returns:
    numpy.ndarray: The translated image.
    """
    # Get the number of rows and columns in the image
    rows, cols = image.shape
    
    # Define the translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Perform the translation
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    
    return translated_image

# Only use when you have a good image to calibrate
def align_images(ref_image, image_to_align):
    # Debugging: Print image shapes and types
    print("Reference image shape:", ref_image.shape, "dtype:", ref_image.dtype)
    print("Image to align shape:", image_to_align.shape, "dtype:", image_to_align.dtype)

    # Ensure images are in uint8 format for ORB (converting if necessary)
    if ref_image.dtype != np.uint8:
        ref_image = cv2.convertScaleAbs(ref_image, alpha=(255.0/65535.0))
    if image_to_align.dtype != np.uint8:
        image_to_align = cv2.convertScaleAbs(image_to_align, alpha=(255.0/65535.0))

    # Convert images to grayscale if they are not already
    if len(ref_image.shape) == 3 and ref_image.shape[2] > 1:
        ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    else:
        ref_image_gray = ref_image

    if len(image_to_align.shape) == 3 and image_to_align.shape[2] > 1:
        image_to_align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    else:
        image_to_align_gray = image_to_align

    # Debugging: Print image shapes after conversion
    print("Grayscale reference image shape:", ref_image_gray.shape)
    print("Grayscale image to align shape:", image_to_align_gray.shape)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(ref_image_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image_to_align_gray, None)

    if descriptors1 is None or descriptors2 is None:
        print("Error: No descriptors found in one or both images.")
        return None

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) == 0:
        print("Error: No matches found.")
        return None

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Compute homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    if h is None:
        print("Error: Homography could not be computed.")
        return None

    # Use homography to warp image
    height, width = ref_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, h, (width, height))

    return aligned_image

def calculate_ndvi(nir_path, red_path, save_path):
    with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
        nir = nir_src.read(1).astype(float)
        red = red_src.read(1).astype(float)

        # Avoid division by zero
        np.seterr(divide='ignore', invalid='ignore')

        # Calculate NDVI
        ndvi = (nir - red) / (nir + red)

        # Define meta data for output file based on the NIR band
        meta = nir_src.meta
        meta.update(dtype=rasterio.float32, count=1, compress='lzw')

        aligned_path = save_path.replace('_RED_ALIGNED.TIF', '_NDVI.TIF')
        print(aligned_path)
        # Write NDVI image
        try:
            with rasterio.open(aligned_path, 'w', **meta) as ndvi_dst:
                ndvi_dst.write(ndvi.astype(rasterio.float32), 1)
            print(f"Successfully wrote NDVI image to {aligned_path}")
        except rasterio.errors.RasterioIOError as e:
            print(f"Rasterio error: {e}")
        except Exception as e:
            print(f"Error opening file with rasterio: {e}")

        # Create a color map from red to yellow to green
        colormap = cm.ScalarMappable(Normalize(vmin=-1, vmax=1), cm.RdYlGn)

        # Apply the colormap to the NDVI data
        ndvi_colored = colormap.to_rgba(ndvi, bytes=True)

        # Convert colored NDVI to an image array and save
        plt.imsave(save_path.replace('_RED_ALIGNED.TIF', '_NDVI_RGB.png'), ndvi_colored, format='png', dpi=300)
    
    return aligned_path


def process_folder(process_base_folder, process_folder_list, H_RED,  H_NIR2RGB):

    for folder in process_folder_list:
        input_directory = os.path.join(process_base_folder, folder)
        for subfolder in os.listdir(input_directory):
            subfolder_path = os.path.join(input_directory, subfolder)
            image_list_file = os.path.join(subfolder_path, "imageList.txt")
            if os.path.isfile(image_list_file):
                print("found")
                with open(image_list_file, 'r') as file:
                    fulpath_images_list = [image.strip() for image in file.readlines()]
                    
                    for image_file in tqdm(fulpath_images_list ):
                        image_file_path = os.path.join(subfolder_path, image_file)
                        print(image_file_path," ", os.path.isfile(image_file_path))
        
                        nir_image_path = image_file_path.replace("_RGB.png", "_NIR.TIF")
                        red_image_path =image_file_path.replace("_RGB.png", "_RED.TIF")
                        
                        # "NIR", "RED","REG","GRE" 
                        nir_image_ref = tiff.imread(nir_image_path)
                        red_image = tiff.imread(red_image_path)
                        
                        """
                        if nir_image_ref is not None and red_image is not None:
                            aligned_red_image = align_images(nir_image_ref, red_image)
                            if aligned_red_image is not None:
                                print("Images aligned successfully.")
                                
                                # Only do this using the calibration tragets
                                aligned_image_path = os.path.join(subfolder_path, image_file.replace("_RGB.png", "_ALIGNED_RED.TIF"))
                                tiff.imwrite(aligned_image_path, aligned_red_image)
                                print(aligned_image_path)
                            else:
                                print("Alignment failed.")
                        else:
                            print("Error loading images for alignment.")
                        """
                               # Save the aligned image (optional)
                        aligned_image_path = os.path.join(subfolder_path, image_file.replace("_RGB.png", "_ALIGNED_RED.TIF"))

                        # "NIR", "RED","REG","GRE"
                        nir_image_ref = tiff.imread(nir_image_path)
                        red_image = tiff.imread(red_image_path)

                        if nir_image_ref is not None and red_image is not None:
                            
                            # red_image_aligned = align_images(nir_image_ref, red_image)
                            # Hard coded values after at 1 meter off the ground, very good
                            #red_image_aligned  = translate_image(red_image , 5, 25)
                            
                            # Align image with calibration
                            height, width = red_image.shape[:2]
                            red_image_aligned = cv2.warpPerspective(red_image, H_RED, (width, height))
                            
                            
                            if red_image_aligned  is not None:
                                print("Images aligned successfully.")

                                nir_image_aligned_path = image_file_path.replace("_RGB.png", "_NIR_REF.TIF")
                                red_image_aligned_path = image_file_path.replace("_RGB.png", "_RED_ALIGNED.TIF")

                                # tiff.imwrite(nir_image_aligned_path, nir_image_ref)
                                tiff.imwrite(red_image_aligned_path, red_image_aligned)

                                # Calculate and save NDVI
                                
                                ndvi_path = calculate_ndvi(nir_image_aligned_path, red_image_aligned_path, red_image_aligned_path )
                                
                                # Do correction
                                nir_image_ref = tiff.imread(ndvi_path)
                                # Align image with calibration
                                height, width = nir_image_ref.shape[:2]
                                ndvi_image_aligned = cv2.warpPerspective(nir_image_ref,  H_NIR2RGB, (width, height))
                                tiff.imwrite(ndvi_path.replace('_NDVI.TIF', '_NDVI_ALIGNED.TIF'), ndvi_image_aligned)
                                
                            else:
                                print("Alignment failed.")
                        else:
                            print("Error loading images for alignment.", nir_image_path)
                        
                        
                        
                        
        



def main():
    parser = argparse.ArgumentParser(description="Align images ")
    parser.add_argument("--process_folders", type=str, default=config.FOLDER_CAPTURES, help="Input folder names list")

    parser.add_argument("--process", type=str, default=config.BASE_PROCESSED_FOLDER, help="Output folder for rectified images.")

    args = parser.parse_args()
    
    H_RED2NIR = util.load_homography_matrix(config.CALILBRATION_RED2NIR_JSON)
    H_NIR2RGB = util.load_homography_matrix(config.CALILBRATION_NIR2RGB_JSON)
    
    process_folder(args.process, args.process_folders, H_RED2NIR,  H_NIR2RGB )

if __name__ == "__main__":
    main()
    
