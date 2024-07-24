import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

def read_image_without_orientation(path):
    # Open the image with Pillow to access EXIF metadata
    image = Image.open(path)
    

    # Convert Pillow image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    return image_cv

def convert_to_8bit(image):
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))
    elif image.dtype == np.float32 or image.dtype == np.float64:
        image = cv2.convertScaleAbs(image, alpha=255.0)
    return image

def calculate_fractional_green_canopy(image_path, r_to_g_threshold=0.95, b_to_g_threshold=0.95, exg_threshold=20):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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
   # nir_image = cv2.cvtColor(nir_image, cv2.COLOR_GRAY2RGB)
    
    # Save the mask as a grayscale image
    grayscale_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)  # Create a blank mask with the same height and width as the input image
    grayscale_mask[mask] = 255
    cv2.imwrite(output_path, grayscale_mask)
    
    # Draw a 224x224 red rectangle on the overlay


    h, w, _ = overlay.shape
    rect_start = (224, 224)  # Top-left corner
    rect_end = (1120, 896)  # Bottom-right corner
    
    # Draw the rectangle on the original image
    cv2.rectangle(image, rect_start, rect_end, (255, 255, 0), 8)  # Red color with thickness 2
    cv2.rectangle(nir_image, rect_start, rect_end, (255, 255, 0),8)  # Red color with thickness 2
    cv2.rectangle(overlay, rect_start, rect_end, (255, 255, 0), 8)  # Red color with thickness 2
    cv2.rectangle(overlay, rect_start, rect_end, (255, 0, 0),8)  # Red color with thickness 2

    cv2.rectangle(nir_image, rect_start, rect_end, (255, 0, 0),8)  # Red color with thickness 2
    
    if 1 :
        h, w, _ = overlay.shape
        step_size = 224
        
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                rect_start = (x, y)
                rect_end = (x + step_size, y + step_size)
                
                cv2.rectangle(image, rect_start, rect_end, (255, 0, 0), 2)
                cv2.rectangle(nir_image, rect_start, rect_end, (255, 0, 0), 2)
                cv2.rectangle(overlay, rect_start, rect_end, (255, 0, 0), 2)
                
                label = f"{rect_start[0]},{rect_start[1]}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                cv2.putText(nir_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        
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
    ndvi_normalized = cv2.normalize(nir_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
    plt.imshow(ndvi_colored)

    plt.tight_layout()
    plt.show()

# Example usage
img_path = r"D:\datasets\Processed\ATU_01_MAY_2024\0007\IMG_240501_160005_0003_RGB.png"
nir_img_path = r"D:\datasets\Processed\ATU_01_MAY_2024\0007\IMG_240501_160005_0003_NDVI_ALIGNED.TIF"
output_path = r"C:\Users\stevf\OneDrive\Documents\Projects\Github_IMVIP2024\turfgrass_sequoia\test\green_canopy_mask.png"

image, mask, fractional_green_canopy = calculate_fractional_green_canopy(img_path)
visualize_and_save_fractional_green_canopy(image, nir_img_path, mask, fractional_green_canopy, output_path)
