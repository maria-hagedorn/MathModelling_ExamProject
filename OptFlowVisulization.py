import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_data(pickle_file):
    with open(pickle_file, "rb") as file:
        data = pickle.load(file)
    return data

def visualize_optical_flow(flow):
    # Assuming 'flow' is a two-channel array with optical flow vectors
    if flow is not None:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        plt.imshow(bgr)
        plt.title('Optical Flow')
        plt.show()
    else:
        print("No optical flow data available.")

# Load your saved pickle file
data = load_data('all_days_flow_data.pkl')

# Iterate through the saved data structure and visualize the first set of optical flow data
for df in data:
    if 'flow_images' in df.columns and len(df['flow_images']) > 1 and df['flow_images'][1] is not None:
        visualize_optical_flow(df['flow_images'][1])  # Visualize the first non-None flow
        break
