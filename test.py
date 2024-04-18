import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image_array(file_path):
    """Load an image array from a .npy file."""
    return np.load(file_path)

# Path to the .npy image file
file_path = 'data/20240317/MSG3-SEVI-MSG15-0100-NA-20240317064241.584000000Z-NA_natural_color.npy'

# Load the image array
image_array = load_image_array(file_path)

# Check if the image array is RGB
if image_array.ndim == 3 and image_array.shape[2] == 3:
    print('Image array is RGB')
else:
    print('Image array is not RGB')