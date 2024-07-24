#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2. Align the RED LENS to the NIR Channel save, additionally calculate NDVI value

@author: stevefoy
"""

import os
import argparse
import cv2
import numpy as np
import tifffile as tiff
import rasterio
from rasterio.errors import RasterioIOError
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import config
import util

def translate_image(image, tx, ty):
    rows, cols = image.shape
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def align_images(ref_image, image_to_align):
    if ref_image.dtype != np.uint8:
        ref_image = cv2.convertScaleAbs(ref_image, alpha=(255.0 / 65535.0))
    if image_to_align.dtype != np.uint8:
        image_to_align = cv2.convertScaleAbs(image_to_align, alpha=(255.0 / 65535.0))

    if len(ref_image.shape) == 3:
        ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    else:
        ref_image_gray = ref_image

    if len(image_to_align.shape) == 3:
        image_to_align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    else:
        image_to_align_gray = image_to_align

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(ref_image_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image_to_align_gray, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) == 0:
        return None

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    if h is None:
        return None

    height, width = ref_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, h, (width, height))
    return aligned_image

def calculate_ndvi(nir_path, red_path, save_path):
    with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
        nir = nir_src.read(1).astype(float)
        red = red_src.read(1).astype(float)

        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)

        meta = nir_src.meta
        meta.update(dtype=rasterio.float32, count=1, compress='lzw')

        aligned_path = save_path.replace('_RED_ALIGNED.TIF', '_NDVI.TIF')
        try:
            with rasterio.open(aligned_path, 'w', **meta) as ndvi_dst:
                ndvi_dst.write(ndvi.astype(rasterio.float32), 1)
            #print(f"Successfully wrote NDVI image to {aligned_path}")
        except RasterioIOError as e:
            print(f"Rasterio error: {e}")
        except Exception as e:
            print(f"Error opening file with rasterio: {e}")

        colormap = cm.ScalarMappable(Normalize(vmin=-1, vmax=1), cm.RdYlGn)
        ndvi_colored = colormap.to_rgba(ndvi, bytes=True)
        plt.imsave(save_path.replace('_RED_ALIGNED.TIF', '_NDVI_RGB.png'), ndvi_colored, format='png', dpi=300)

    return aligned_path

def process_image_files(image_file_path, subfolder_path, H_RED, H_NIR2RGB):
    nir_image_path = image_file_path.replace("_RGB.png", "_NIR.TIF")
    red_image_path = image_file_path.replace("_RGB.png", "_RED.TIF")



        
    nir_image_ref = tiff.imread(nir_image_path)
    red_image = tiff.imread(red_image_path)

    if nir_image_ref is not None and red_image is not None:
        height, width = red_image.shape[:2]
        red_image_aligned = cv2.warpPerspective(red_image, H_RED, (width, height))

        if red_image_aligned is not None:
            # print("Images aligned successfully.")
            red_image_aligned_path = image_file_path.replace("_RGB.png", "_RED_ALIGNED.TIF")
            tiff.imwrite(red_image_aligned_path, red_image_aligned)

            ndvi_path = calculate_ndvi(nir_image_path, red_image_aligned_path, red_image_aligned_path)
            nir_image_ref = tiff.imread(ndvi_path)
            height, width = nir_image_ref.shape[:2]
            ndvi_image_aligned = cv2.warpPerspective(nir_image_ref, H_NIR2RGB, (width, height))
            tiff.imwrite(ndvi_path.replace('_NDVI.TIF', '_NDVI_ALIGNED.TIF'), ndvi_image_aligned)
        else:
            print("Alignment failed, File: ", image_file_path )
    else:
        print("Error loading images for alignment, File: ", image_file_path)

def process_folder(process_base_folder, process_folder_list, H_RED, H_NIR2RGB):
    for folder in process_folder_list:
        input_directory = os.path.join(process_base_folder, folder)
        for subfolder in os.listdir(input_directory):
            subfolder_path = os.path.join(input_directory, subfolder)
            image_list_file = os.path.join(subfolder_path, "imageList.txt")
            if os.path.isfile(image_list_file):
                print("found")
                with open(image_list_file, 'r') as file:
                    fulpath_images_list = [image.strip() for image in file.readlines()]

                    # Thread pool is class but any files missing it just skipsssssss
                    #with ThreadPoolExecutor() as executor:
                    for image_file in tqdm(fulpath_images_list):
                        print(image_file)
                        image_file_path = os.path.join(subfolder_path, image_file)
                        #executor.submit(process_image_files, image_file_path, subfolder_path, H_RED, H_NIR2RGB)
                        process_image_files( image_file_path, subfolder_path, H_RED, H_NIR2RGB)
def main():
    parser = argparse.ArgumentParser(description="Align images")
    parser.add_argument("--process_folders", type=str, default=config.FOLDER_CAPTURES, help="Input folder names list")
    parser.add_argument("--process", type=str, default=config.BASE_PROCESSED_FOLDER, help="Output folder for rectified images.")
    args = parser.parse_args()

    H_RED2NIR = util.load_homography_matrix(config.CALILBRATION_RED2NIR_JSON)
    H_NIR2RGB = util.load_homography_matrix(config.CALILBRATION_NIR2RGB_JSON)

    process_folder(args.process, args.process_folders, H_RED2NIR, H_NIR2RGB)

if __name__ == "__main__":
    main()
