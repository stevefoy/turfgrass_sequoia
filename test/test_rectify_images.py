import pytest
import numpy as np
import cv2
from src.rectify_images import rectify_image

# Sample test data for rectification
# In practice, use real data or mock data close to your use case
def generate_mock_image():
    """Generate a mock image for testing."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 25, (255, 255, 255), -1)
    return image

def generate_mock_calibration_data():
    """Generate mock calibration data for testing."""
    return {
        'camera_matrix': np.array([[1000, 0, 50], [0, 1000, 50], [0, 0, 1]]),
        'dist_coeffs': np.array([0.1, -0.1, 0, 0, 0])
    }

@pytest.fixture
def mock_image():
    return generate_mock_image()

@pytest.fixture
def mock_calibration_data():
    return generate_mock_calibration_data()

def test_rectify_image(mock_image, mock_calibration_data):
    """Test the image rectification function."""
    rectified_image = rectify_image(mock_image, mock_calibration_data)
    
    # Check that the rectified image is still the same size
    assert rectified_image.shape == mock_image.shape, "Rectified image has a different shape"

    # Check that the rectified image is not just the same as the input (indicating a change)
    assert not np.array_equal(rectified_image, mock_image), "Rectified image is the same as input image"

    # Additional checks can be added here based on expected outcomes

if __name__ == "__main__":
    pytest.main([__file__])

