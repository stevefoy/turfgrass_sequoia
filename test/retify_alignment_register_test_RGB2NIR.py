import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def enhance_image(image):
    """Enhance the image for better checkerboard detection."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def preprocess_nir_image(image):
    """Preprocess the NIR image to enhance checkerboard detection."""
    # Normalize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Apply Gaussian Blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
    return image

def apply_roi(image, roi):
    """Apply a region of interest to the image."""
    x, y, w, h = roi
    return image[y:y+h, x:x+w]

def find_checkerboard_corners(image, pattern_size, is_nir=False):
    """Find checkerboard corners in the image."""
    if is_nir:
        image = preprocess_nir_image(image)
    else:
        image = enhance_image(image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return ret, corners, gray

def align_images(reference_image, image_to_align, corners_ref, corners_align):
    """Align image_to_align to reference_image using the checkerboard corners."""
    H, _ = cv2.findHomography(corners_align, corners_ref, cv2.RANSAC, 5.0)
    height, width = reference_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, H, (width, height))
    return aligned_image, H

def calculate_reprojection_error(corners_ref, corners_proj):
    """Calculate the re-projection error."""
    errors = np.sqrt(np.sum((corners_ref - corners_proj) ** 2, axis=2))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    return mean_error, max_error, errors

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

def decompose_homography(H):
    """Decompose the homography matrix into scale, rotation, and translation components."""
    H = H / H[2, 2]
    norm1 = np.linalg.norm(H[:, 0])
    norm2 = np.linalg.norm(H[:, 1])
    scale_x = norm1
    scale_y = norm2
    rotation = H[:, :2] / np.array([norm1, norm2])
    translation = H[:, 2]
    theta = np.arctan2(rotation[1, 0], rotation[0, 0]) * 180 / np.pi

    return scale_x, scale_y, theta, translation

def main(nir_img_path, rgb_img_path, pattern_size, roi_nir, roi_rgb, homography_file):
    nir_img = load_image(nir_img_path)
    rgb_img = load_image(rgb_img_path)

    print(f"Loaded NIR image shape: {nir_img.shape}")
    print(f"Loaded RGB image shape: {rgb_img.shape}")

    # Apply the ROI to both images
    nir_img = apply_roi(nir_img, roi_nir)
    rgb_img = apply_roi(rgb_img, roi_rgb)

    # Debug: visualize the ROI applied images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('NIR Image with ROI')
    plt.imshow(cv2.cvtColor(nir_img, cv2.COLOR_BGR2RGB) if len(nir_img.shape) == 3 else nir_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('RGB Image with ROI')
    plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) if len(rgb_img.shape) == 3 else rgb_img, cmap='gray')
    plt.show()

    # Find checkerboard corners
    ret_nir, corners_nir, gray_nir = find_checkerboard_corners(nir_img, pattern_size, is_nir=True)
    ret_rgb, corners_rgb, gray_rgb = find_checkerboard_corners(rgb_img, pattern_size, is_nir=False)

    # Debug: visualize checkerboard detection
    if ret_nir:
        cv2.drawChessboardCorners(nir_img, pattern_size, corners_nir, ret_nir)
    if ret_rgb:
        cv2.drawChessboardCorners(rgb_img, pattern_size, corners_rgb, ret_rgb)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Checkerboard Detection - NIR Image')
    plt.imshow(cv2.cvtColor(nir_img, cv2.COLOR_BGR2RGB) if len(nir_img.shape) == 3 else nir_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Checkerboard Detection - RGB Image')
    plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) if len(rgb_img.shape) == 3 else rgb_img, cmap='gray')
    plt.show()

    if not ret_nir or not ret_rgb:
        print("Checkerboard corners not found in one or both images.")
        return

    # Align the RGB image to the NIR image
    aligned_rgb_img, H = align_images(nir_img, rgb_img, corners_nir, corners_rgb)
    
    # Save the homography matrix
    save_homography_matrix(H, homography_file)
    print(f"Saved homography matrix to {homography_file}")

    # Decompose the homography matrix
    scale_x, scale_y, theta, translation = decompose_homography(H)
    print(f"Scale X: {scale_x}, Scale Y: {scale_y}")
    print(f"Rotation (degrees): {theta}")
    print(f"Translation: {translation}")

    # Ensure both images are the same size and number of channels for absdiff
    if aligned_rgb_img.shape != nir_img.shape:
        aligned_rgb_img = cv2.resize(aligned_rgb_img, (nir_img.shape[1], nir_img.shape[0]))

    if len(aligned_rgb_img.shape) == 2:
        aligned_rgb_img = cv2.cvtColor(aligned_rgb_img, cv2.COLOR_GRAY2BGR)
    if len(nir_img.shape) == 2:
        nir_img = cv2.cvtColor(nir_img, cv2.COLOR_GRAY2BGR)
    
    # Calculate the absolute difference between the two images
    diff_image = cv2.absdiff(nir_img, aligned_rgb_img)
    diff_image_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
    _, diff_image_thresh = cv2.threshold(diff_image_gray, 30, 255, cv2.THRESH_BINARY)

    # Calculate the mean of the differences
    mean_diff = np.mean(diff_image_thresh)
    print(f'Mean pixel difference between images: {mean_diff}')

    # Calculate re-projection error
    corners_rgb_proj = cv2.perspectiveTransform(corners_rgb, H)
    mean_reproj_error, max_reproj_error, reproj_errors = calculate_reprojection_error(corners_nir, corners_rgb_proj)
    print(f'Mean re-projection error: {mean_reproj_error}')
    print(f'Maximum re-projection error: {max_reproj_error}')

    # Visualize the results
    plt.figure(figsize=(24, 12))
    plt.subplot(2, 2, 1)
    plt.title('NIR Image')
    plt.imshow(cv2.cvtColor(nir_img, cv2.COLOR_BGR2RGB) if len(nir_img.shape) == 3 else nir_img, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('RGB Image (Aligned)')
    plt.imshow(cv2.cvtColor(aligned_rgb_img, cv2.COLOR_BGR2RGB) if len(aligned_rgb_img.shape) == 3 else aligned_rgb_img, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('Absolute Difference')
    plt.imshow(diff_image_thresh, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Re-projection Errors')
    plt.scatter(range(len(reproj_errors)), reproj_errors, c='r')
    plt.xlabel('Point Index')
    plt.ylabel('Re-projection Error (pixels)')
    plt.grid(True)
    
    plt.show()

def apply_homography_to_folder(homography_file, input_folder, output_folder, roi):
    """Apply the saved homography matrix to all images in the input folder."""
    H = load_homography_matrix(homography_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = load_image(img_path)

        # Apply ROI
        img = apply_roi(img, roi)
        
        # Align image
        height, width = img.shape[:2]
        aligned_img = cv2.warpPerspective(img, H, (width, height))

        # Save aligned image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, aligned_img)
        print(f"Saved aligned image to {output_path}")

if __name__ == "__main__":
    nir_img_path = '/media/freddy/vault/datasets/Parrot/zendo_release/processed/Calibration/0020/IMG_240711_094018_0000_NIR.TIF'
    rgb_img_path = '/media/freddy/vault/datasets/Parrot/zendo_release/processed/Calibration/0020/IMG_240711_094018_0000_RGB.png'
    pattern_size = (12, 8)
    roi_nir = (0, 25, 1280, 960)  # Example ROI coordinates (x, y, width, height)
    roi_rgb = (0, 25, 1280, 960)  # Example ROI coordinates (x, y, width, height)
    homography_file = 'align_RGB2NIR.json'
    main(nir_img_path, rgb_img_path, pattern_size, roi_nir, roi_rgb, homography_file)

    # Apply homography to a folder of images
    #input_folder = '/path/to/rgb/images'
    #output_folder = '/path/to/aligned/images'
    #apply_homography_to_folder(homography_file, input_folder, output_folder, roi_rgb)
