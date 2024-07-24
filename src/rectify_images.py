#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. Process the images 

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

colors = list(mcolors.TABLEAU_COLORS.values())

# Function to undistort image with perspective distortion
def undistort_image_perspective(focal_length_mm, dist_coeffs, image, sensor_width_mm, sensor_height_mm):

    # Given parameters
    principal_point = (3.076679, 2.275708)  # (cx, cy)
    focal_plane_x_resolution = 746.2686793
    focal_plane_y_resolution = 746.2686793

    # Get image dimensions
    h, w = image.shape[:2]

    # Convert focal length to pixels
    focal_length_px = (focal_length_mm * w) / sensor_width_mm

    # Assuming the principal point is at the center of the image
    cx = principal_point[0] * focal_plane_x_resolution
    cy = principal_point[1] * focal_plane_y_resolution

    # Construct the camera matrix
    cam_mat = np.array([[focal_length_px, 0, cx],
                        [0, focal_length_px, cy],
                        [0, 0, 1]])

    #print("Camera Matrix:\n", cam_mat)
    #print("Distortion Coefficients:\n", dist_coeffs)

    # Obtain the optimal new camera matrix
    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w, h), 1, (w, h))

    #print("New Camera Matrix:\n", new_cam_mat)
    #print("ROI:\n", roi)

    # Undistort the image using the optimal new camera matrix
    undistorted_image = cv2.undistort(image, cam_mat, dist_coeffs, None, new_cam_mat)

    if undistorted_image is None or undistorted_image.size == 0:
        print("Error: Undistorted image is empty")
    else:
        # Crop the image based on the roi (region of interest)
        x, y, w, h = roi
        # print("ROI", roi)
        undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image


def process_RGB_image(imageRaw, filepath, save_path):

    metaToolData = util.process_metadata(filepath)
    
                      # Extract the focal plane resolution
    focal_plane_x_res = metaToolData.get('EXIF:FocalPlaneXResolution')
    focal_plane_y_res = metaToolData.get('EXIF:FocalPlaneYResolution')
    
    pitch_angle = metaToolData.get('XMP:Pitch')
    roll_angle = metaToolData.get('XMP:Roll')
    yaw_angle = metaToolData.get('XMP:Yaw')
    print("Angles", pitch_angle," ", roll_angle," ", yaw_angle)
    
                        # Calculate sensor size
    sensor_width_mm = 4000 / focal_plane_x_res
    sensor_height_mm = 3000 / focal_plane_y_res
    ##print(sensor_width_mm,  sensor_height_mm)
    #print("Sensor Width (mm):", sensor_width_mm)
   # print("Sensor Height (mm):", sensor_height_mm)
    
    # Distortion coefficients for opencv: [R1, R2, T1, T2, R3]
    # Specific focal length and distortion coefficients from metadata
    focal_length_mm = 4.825853
    # Distortion coefficients, order: [R1, R2, T1, T2, R3]
    dist_coeffs = np.array([0.180417518, -0.468309449, -0.000115991, 0.000489171, 0.376562436])
    
    # Correct perspective distortion
    undistorted_image_rgb = undistort_image_perspective( focal_length_mm, dist_coeffs, imageRaw, sensor_width_mm, sensor_height_mm)
    undistorted_image_rgb = cv2.resize( undistorted_image_rgb , (1280, 960))
    
    
    cv2.imwrite(save_path, undistorted_image_rgb)


def process_folder(input_baser_folder, input_folder_list, output_folder):
    process_RGB = True
    
    image_count_folder = 0
    for i, folder in enumerate(input_folder_list):
        print("Folder: ", folder)
        if i > len(colors):
            ix = i % len(colors)
            color = colors[ix]
        
        folder_path = os.path.join(input_baser_folder, folder)
        
        for subdir, dirs, files in os.walk(folder_path):
            
            # Deleted in raw but just in case
            # Ignore the .thumb directory
            if '.thumb' in dirs:
                dirs.remove('.thumb')
            # Ignore the .thumb directory
            if 'IMG.thumb' in dirs:
                dirs.remove('IMG.thumb')  
            
            timeDelta = 0
            #for file in tqdm(files):
            for file in files:
                filepath = os.path.join(subdir, file)
                save_subdir = subdir.replace(input_baser_folder, output_folder)
                
                # save all the files in the same directory
                #save_subdir = os.path.join(save_folder_path, folder)
                    
                os.makedirs(save_subdir, exist_ok=True)
                
                if file.endswith("_RGB.JPG") :
                    image_count_folder +=1
                    
                #process RGB sensor
                if file.endswith("_RGB.JPG") :
                    
                    import time
                    start = time.time()
                    imageRaw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    save_path = os.path.join(save_subdir, file.replace(".JPG", ".png"))
                    
                    process_RGB_image(imageRaw, filepath, save_path)
                    print(save_path)
                    end = time.time()
                    timeDelta = end - start
                    process_RGB = False
                
                    process_bands= True
                    if process_bands == True:
                        # NIR
                        #filepath = os.path.join(subdir, file)
                        #save_subdir = subdir.replace(base_folder_path, save_folder_path)
                        #TypeFiles = ["NIR", "RED","REG","GRE" ]
                        TypeFiles = ["NIR", "RED" ]
                        TypesNames = []
                        for extTypes in TypeFiles:
                            related_files = util.find_related_files(file, subdir, extTypes)
                            #print(f"RGB file: {file}")
                            # print("Related files:", related_files)
                            filepath = os.path.join(subdir, related_files[0])
                            TypesNames.append(filepath )
                        print("Related files:", len(TypesNames))
                        
                        for filepathTIF in TypesNames:
                            imageRawTIF = cv2.imread(filepathTIF, cv2.IMREAD_UNCHANGED)
                            # Known issue with open raw and flip issus if exif has 1 for flip
                            imageRaw = cv2.rotate(imageRawTIF, cv2.ROTATE_180)
                            
                            metaToolData = util.process_metadata(filepathTIF)
                            
                            # Try image undistort direct
                            # undistortedImage = util_sequoi.correct_lens_distortion_sequoia(metaToolData, imageRawTIF )
                            
                            # 1. Function does sequoia_vignetteCorrection
                            image_vignette_correction = util_sequoi.sequoia_vignetteCorrection(metaToolData, imageRaw)
                            # 2. image vignette_correction data into irradiance values, specific formula in OEM sheet
                            image_irradiance = util_sequoi.sequoia_irradiance(metaToolData, image_vignette_correction)
                            # 3. sunshine sensor Irradiance stored from calibratoin 
                            sunsenor_irradiance_value = util_sequoi.sunIrradiance(metaToolData)
                            # 4. light calibrated sequoia irradiance formula
                            irradiance_calibrated = image_irradiance/sunsenor_irradiance_value
                            # 5. correct for lens distortions 
                            undistorted_image  = util_sequoi.correct_lens_distortion_sequoia(metaToolData, irradiance_calibrated)
                            # 6. Normalize if necessary (e.g., if values are not already in [0, 1])
                            undistorted_image = undistorted_image  / undistorted_image.max()
                            # 7. Scale to uint16 range 
                            image_uint16 = (undistorted_image * 65535).astype(np.uint16)
                            #8. Finally Save
                            save_path = filepathTIF.replace(input_baser_folder, output_folder)
                            
                            tiff.imwrite( save_path, image_uint16)
                        
    print("Total processed:",timeDelta," ", image_count_folder*timeDelta)




def main():
    parser = argparse.ArgumentParser(description="Rectify images using calibration data.")
    parser.add_argument("--input", type=str, default=config.BASE_INPUT_FOLDER, help="Input folder containing images.")
    parser.add_argument("--input_folders", type=str, default=config.FOLDER_CAPTURES, help="Input folder names list")

    parser.add_argument("--output", type=str, default=config.BASE_PROCESSED_FOLDER, help="Output folder for rectified images.")

    args = parser.parse_args()

    process_folder(args.input,args.input_folders , args.output)

if __name__ == "__main__":
    main()
    
