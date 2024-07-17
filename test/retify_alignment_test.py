import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def preprocess_image(image):
    """Convert image to grayscale if needed and apply Canny edge detection."""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def find_checkerboard_contours(image):
    """Find contours in the image using edge detection."""
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours, min_area=200):
    """Filter out small contours based on a minimum area threshold."""
    return [c for c in contours if cv2.contourArea(c) > min_area]

def cluster_contours(contours, n_clusters=2):
    """Cluster contour points to estimate the number of clusters (pattern size)."""
    points = []
    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        points.append([cX, cY])
    
    points = np.array(points)
    clustering = KMeans(n_clusters=n_clusters).fit(points)
    row_labels = clustering.labels_
    
    unique_labels = np.unique(row_labels)
    num_clusters = len(unique_labels)
    
    if num_clusters > 1:
        return num_clusters
    return None

def detect_and_draw_corners(image, pattern_size):
    """Detect and draw checkerboard corners on the image."""
    ret, corners = cv2.findChessboardCorners(image, pattern_size)
    if ret:
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    return ret, corners, image

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

def detect_lines(image):
    """Detect lines in the edge-detected image using Hough Transform."""
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    return lines

def measure_lines(lines):
    """Measure the lengths and angles of the detected lines."""
    lengths = []
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            lengths.append(length)
            angles.append(angle)
    return lengths, angles

def main(before_img_path, after_img_path, roi, roi2):
    before_img = load_image(before_img_path)
    after_img = load_image(after_img_path)

    # Apply the ROI to both images
    #before_img = apply_roi(before_img, roi)
    #after_img = apply_roi(after_img, roi2)

    # Preprocess the images
    edges_before = preprocess_image(before_img)
    edges_after = preprocess_image(after_img)

    # Detect lines in both images
    lines_before = detect_lines(edges_before)
    lines_after = detect_lines(edges_after)

    # Measure lines in both images
    lengths_before, angles_before = measure_lines(lines_before)
    lengths_after, angles_after = measure_lines(lines_after)

    # Calculate mean length and angle differences
    mean_length_before = np.mean(lengths_before)
    mean_length_after = np.mean(lengths_after)
    mean_angle_before = np.mean(angles_before)
    mean_angle_after = np.mean(angles_after)

    length_diff = mean_length_after - mean_length_before
    angle_diff = mean_angle_after - mean_angle_before

    print(f'Mean line length before calibration: {mean_length_before}')
    print(f'Mean line length after calibration: {mean_length_after}')
    print(f'Mean angle before calibration: {mean_angle_before}')
    print(f'Mean angle after calibration: {mean_angle_after}')
    print(f'Difference in line lengths: {length_diff}')
    print(f'Difference in angles: {angle_diff}')

    # Visualize the results
    plt.figure(figsize=(24, 12))
    plt.subplot(2, 2, 1)
    plt.title('Edges Before Rectification')
    plt.imshow(edges_before, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Edges After Rectification')
    plt.imshow(edges_after, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('Lines Before Rectification')
    if lines_before is not None:
        for line in lines_before:
            for x1, y1, x2, y2 in line:
                plt.plot([x1, x2], [y1, y2], 'r')
    plt.imshow(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 4)
    plt.title('Lines After Rectification')
    if lines_after is not None:
        for line in lines_after:
            for x1, y1, x2, y2 in line:
                plt.plot([x1, x2], [y1, y2], 'r')
    plt.imshow(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))

    plt.show()

if __name__ == "__main__":
    before_img_path = '/media/freddy/vault/datasets/Parrot/Calibration_tests/fulltarget_0020/IMG_240711_094018_0000_NIR.TIF'
    after_img_path = '/media/freddy/vault/datasets/Parrot/Calibration_tests/Processed/fulltarget_0020/IMG_240711_094018_0000_NIR.TIF'
    roi = (120, 120, 1100, 740)  # Example ROI coordinates (x, y, width, height)
    roi2 = (100, 120, 1200, 800)  # Example ROI coordinates (x, y, width, height)
    main(before_img_path, after_img_path, roi, roi2)
