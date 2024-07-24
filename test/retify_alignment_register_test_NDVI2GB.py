import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_fractional_green_canopy(image_path, r_to_g_threshold=0.95, b_to_g_threshold=0.95, exg_threshold=20):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image from path: {image_path}")
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split the image into its R, G, and B components
    R, G, B = cv2.split(image)
    
    # Calculate the R/G and B/G ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        R_to_G = np.true_divide(R, G, where=G != 0)
        B_to_G = np.true_divide(B, G, where=G != 0)
    
    # Calculate the Excess Green Index (ExG)
    ExG = 2 * G - R - B
    
    # Create a mask for green canopy detection based on the thresholds
    mask = (R_to_G < r_to_g_threshold) & (B_to_G < b_to_g_threshold) & (ExG > exg_threshold)
    
    # Calculate the fractional green canopy
    fractional_green_canopy = np.sum(mask) / mask.size
    
    return image, mask, fractional_green_canopy

def align_images(rgb_image, nir_image):
    # Convert images to grayscale
    gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_nir = cv2.normalize(nir_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_rgb, None)
    kp2, des2 = orb.detectAndCompute(gray_nir, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points_rgb = np.zeros((len(matches), 2), dtype=np.float32)
    points_nir = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_rgb[i, :] = kp1[match.queryIdx].pt
        points_nir[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points_nir, points_rgb, cv2.RANSAC)

    # Use homography to warp NIR image
    height, width, _ = rgb_image.shape
    nir_aligned = cv2.warpPerspective(nir_image, h, (width, height))

    return nir_aligned

def visualize_fractional_green_canopy(image_path, nir_img_path, mask, fractional_green_canopy):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image.copy()
    overlay[mask] = [0, 255, 0]  # Green color for the canopy
    
    nir_image = cv2.imread(nir_img_path, cv2.IMREAD_ANYDEPTH)
    if nir_image is None:
        raise ValueError(f"Error loading NIR image from path: {nir_img_path}")
    nir_image = cv2.normalize(nir_image, None, 0, 255, cv2.NORM_MINMAX)
    nir_image = cv2.cvtColor(nir_image, cv2.COLOR_GRAY2RGB)
    
    nir_aligned = align_images(fractional_green_canopy, nir_image)
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    
    plt.subplot(1, 3, 2)
    plt.title('Green Canopy Detection')
    plt.imshow(overlay)
    plt.xlabel(f'Fractional Green Canopy: ')
    
    plt.subplot(1, 3, 3)
    plt.title('Aligned NIR Image')
    plt.imshow(nir_aligned)

    plt.tight_layout()
    plt.show()

# Example usage
img_path = r"D:\datasets\Processed\ATU_30_JAN_2024\0118\IMG_240130_143104_0000_RGB.png"
nir_img_path = r"D:\datasets\Processed\ATU_30_JAN_2024\0118\IMG_240130_143104_0000_NIR.TIF"

mgg, mask, fractional_green_canopy = calculate_fractional_green_canopy(img_path)
visualize_fractional_green_canopy(img_path, nir_img_path, mask, mgg)
