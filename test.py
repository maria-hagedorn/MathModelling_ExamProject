import numpy as np
import cv2
import matplotlib.pyplot as plt

with open("all_days_data.pkl", "rb") as file:
    loaded_data = pickle.load(file)



with open("all_days_flow_data.pkl", "wb") as file:
    pickle.dump(data_optical_flow, file)
