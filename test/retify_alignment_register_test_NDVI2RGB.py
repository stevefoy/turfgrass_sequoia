import cv2
import numpy as np
import matplotlib.pyplot as plt

# Convert image to 8-bit if it's not already
def convert_to_8bit(image):
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))
    elif image.dtype == np.float32 or image.dtype == np.float64:
        image = cv2.convertScaleAbs(image, alpha=255.0)
    return image

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

def visualize_and_save_fractional_green_canopy(image, nir_img_path, mask, fractional_green_canopy, output_path):
    # Create an overlay for the detected green canopy
    overlay = image.copy()
    overlay[mask] = [0, 255, 0]  # Green color for the canopy
    
    # Load and normalize the NIR image
    nir_image = convert_to_8bit(cv2.imread(nir_img_path, cv2.IMREAD_UNCHANGED))
    if nir_image is None:
        raise ValueError(f"Error loading NIR image from path: {nir_img_path}")
    nir_image = cv2.normalize(nir_image, None, 0, 255, cv2.NORM_MINMAX)
    nir_image = cv2.cvtColor(nir_image, cv2.COLOR_GRAY2RGB)
    
    # Save the mask as a grayscale image
    grayscale_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)  # Create a blank mask with the same height and width as the input image
    grayscale_mask[mask] = 255
    cv2.imwrite(output_path, grayscale_mask)
    
    # Create a plot to visualize the images
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('RGB Image')
    plt.imshow(image)
    
    plt.subplot(1, 3, 2)
    plt.title('RGB Green Canopy Detection')
    plt.imshow(overlay)
    plt.xlabel(f'Fractional Green Canopy: {fractional_green_canopy:.2%}')
    
    plt.subplot(1, 3, 3)
    plt.title('NDVI Image')
    plt.imshow(nir_image)

    plt.tight_layout()
    plt.show()

# Example usage
img_path = r"D:\datasets\Processed\ATU_30_JAN_2024\0118\IMG_240130_143117_0009_RGB.png"
nir_img_path = r"D:\datasets\Processed\ATU_30_JAN_2024\0118\IMG_240130_143117_0009_NDVI_ALIGNED.TIF"
output_path = r"C:\Users\stevf\OneDrive\Documents\Projects\Github_IMVIP2024\turfgrass_sequoia\test\green_canopy_mask.png"

image, mask, fractional_green_canopy = calculate_fractional_green_canopy(img_path)
visualize_and_save_fractional_green_canopy(image, nir_img_path, mask, fractional_green_canopy, output_path)
