import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
import pickle

class daily_data_structure:
    def __init__(self, data):
        self.data = data
        self.num_days = len(data)

    def get_day(self, index):
        return self.data[index]
    
def apply_nocloud_filter(rgb_image):
    """Applies the NoCloud filter to an RGB image."""
    R, G, B = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    NoCloud = R + G - 2 * B
    return cv2.normalize(NoCloud, None, 0, 255, cv2.NORM_MINMAX)

def apply_denmark_mask(image, mask_path):
    """Applies the Denmark mask to an image."""
    mask = np.load(mask_path)
    return image * mask

def convert_to_grayscale(image):
    """Converts an image to grayscale, checking if it is RGB."""
    if image.ndim == 3 and image.shape[2] == 3:  # Check if the image is RGB
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:  # The image is already grayscale
        return image
    else:
        raise ValueError("Unexpected image format. Expected RGB or grayscale.")

def calculate_optical_flow(prev_image, current_image):
    """ Calculates and returns the optical flow between two images. """
    return cv2.calcOpticalFlowFarneback(prev_image, current_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

base_dir = 'data'
output_dir = 'data/flows'   # Directory for output flow data
mask_path = 'data/mask.npy' # Denmark mask file path

# Compile a regular expression pattern to match directories like "202403xx"
pattern = re.compile(r"202403\d\d$")

# List and sort directories of the image folders
days = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d)) and pattern.match(d)]

with open("all_days_data.pkl", "rb") as file:
    loaded_data = pickle.load(file)

data_optical_flow = []
for i, day in enumerate(days):
    # Path to the day directory
    day_dir = os.path.join(base_dir, day)
    image_arrays = sorted([f for f in os.listdir(day_dir) if f.endswith('.npy')])

    df = loaded_data.get_day(i)
    df_copy = df.copy()
    
    # Reset the previous image for each day, account for jump between days
    prev_image = None 
    flow_images = []

    for i, file_name in enumerate(image_arrays):
        # Path to the image array file
        image_path = os.path.join(day_dir, file_name)
        i_image = np.load(image_path)

        # Check if the image array is RGB
        if i_image.ndim == 3 and i_image.shape[2] == 3:
            # Apply the NoCloud filter
            nocloud_image = apply_nocloud_filter(i_image)

            # Convert to grayscale
            grayscale_image = convert_to_grayscale(nocloud_image)

            # Apply the Denmark mask
            current_image = apply_denmark_mask(grayscale_image, mask_path)

            if prev_image is not None:
                flow = calculate_optical_flow(prev_image, current_image)
                flow_images.append(flow)
            else:
                flow_images.append(None)

            prev_image = current_image

        else:
            print("The loaded array is not in the expected RGB format.")

    df_copy['Image'] = flow_images
    df_copy.rename(columns={'Image': 'flow_images'}, inplace=True) 
    data_optical_flow.append(df_copy)

    with open("all_days_flow_data.pkl", "wb") as file:
        pickle.dump(data_optical_flow, file)
