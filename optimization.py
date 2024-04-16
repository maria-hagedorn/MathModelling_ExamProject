import pandas as pd
from scipy.optimize import minimize
from PIL import Image
import numpy as np
import os
import re
import pickle
import tifffile as tiff
import matplotlib.pyplot as plt

def loss_function(beta, X, Y, lambdas, Gammas):

    n = int(len(beta) / 2)

    beta1 = beta[:n]
    beta2 = beta[n:]

    #loss = sum([(Y[i] - (X[i][0, :, :]@beta1).T@(X[i][1, :, :]@beta2)).T@(Y[i] - (X[i][0, :, :]@beta1).T@(X[i][1, :, :]@beta2)) for i in range(len(X))]) + lambdas[0]*beta1.T@Gammas[0]@beta1 + lambdas[1]*beta2.T@Gammas[1]@beta2
    loss = sum([Y[i] - (X[i][0, :, :]@beta1).T@(X[i][1, :, :]@beta2) for i in range(len(X))]) + lambdas[0]*beta1.T@Gammas[0]@beta1 + lambdas[1]*beta2.T@Gammas[1]@beta2

    return loss


def train_model(X, Y, loss_function):

    n = X[0].shape[2]

    lambdas = [0.5, 0.5]
    Gammas = [np.eye(n), np.eye(n)]

    initial_guess = np.ones(2*n)

    result = minimize(loss_function,
                      initial_guess,
                      args=(X, Y, lambdas, Gammas),
                      method='SlSQP')

    beta = result.x
    optimal_loss = result.fun

    beta1 = beta[:n]
    beta2 = beta[n:]

    return lambda x: (x[0, :, :]@beta1).T@(x[1, :, :]@beta2), optimal_loss


def extract_date_time_from_filenames(folder_path):

    pattern = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")
    date_time_tuples = []

    for filename in os.listdir(folder_path):
        match = pattern.search(filename)

        if match:
            year, month, day, hour, minute, second = match.groups()
            date_time_tuples.append((f"{year}-{month}-{day}", f"{hour}:{minute}:{second}"))

    return date_time_tuples


def load_satellite_images(folder_path):

    """Returns list of 2-tuples where the first element is the grayscale image
    and the second element is a 2-tuple containing the date and time of the image."""

    pattern = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")
    images_with_date_times = []

    for filename in os.listdir(folder_path):

        match = pattern.search(filename)

        if match and filename.endswith('.tif'):
            year, month, day, hour, minute, second = match.groups()
            date_time = (f"{year}-{month}-{day}", f"{hour}:{minute}:{second}")

            image = tiff.imread(os.path.join(folder_path, filename))

            if image.ndim == 2:
                rgb_image = np.stack([image] * 3, axis=-1)
            elif image.ndim == 3 and image.shape[2] in [3, 4]:
                rgb_image = image[..., :3]
            else:
                rgb_image = image
            rgb_array = np.array(rgb_image)

            R = rgb_array[:, :, 0]
            G = rgb_array[:, :, 1]
            B = rgb_array[:, :, 2]

            grayscale = R + G - 2*B

            # Squeeze the grayscale values to the 0-255 range
            min_val = np.min(grayscale)
            max_val = np.max(grayscale)
            if max_val != min_val:
                grayscale_squeezed = 255 * (grayscale - min_val) / (max_val - min_val)
            else:
                grayscale_squeezed = np.full(grayscale.shape, fill_value=(1-np.round(1/(min_val+0.0001))*min(min_val, 0))*(min(min_val, 255)), dtype=np.uint8)

            # Convert the grayscale array back to an image
            grayscale_image = Image.fromarray(grayscale, 'L')

            images_with_date_times.append((grayscale_image, date_time))

    images = pd.DataFrame({
        'Image': [tup[0] for tup in images_with_date_times],
        'DateTime': [' '.join(tup[1]) for tup in images_with_date_times]  # Combining date and time into a single string
    })
    images['DateTime'] = pd.to_datetime(images['DateTime'])
    images.set_index('DateTime', inplace=True)

    return images


def load_solar_power_generation_data(file_path):
    excel_data = pd.read_csv(file_path, sep=",", parse_dates=['Minutes1UTC'], index_col='Minutes1UTC')
    excel_data = excel_data.drop(columns=['Minutes1DK'])
    solar_power_generation_data = excel_data['SolarPower']
    return solar_power_generation_data


class daily_data_structure:
    def __init__(self, data):
        self.data = data
        self.num_days = len(data)

    def get_day(self, index):
        return self.data[index]


class model:
    def __init__(self, predictor, loss):
        self.loss = loss
        self.predict = predictor


# READ AND SAVE DATA
# all_data = []
#
# for filename in os.listdir('data/csv_files'):
#     satellite_images = load_satellite_images(f'data/{os.path.splitext(filename)[0]}')
#     solar_power_generation_data = load_solar_power_generation_data(f'data/csv_files/{filename}')
#     all_data.append(pd.merge_asof(satellite_images.sort_index(), solar_power_generation_data.sort_index(), left_index=True, right_index=True, direction='backward'))
#
# data = daily_data_structure(all_data)
#
#
# # Serialize the data object to a file
# with open("all_days_data.pkl", "wb") as file:
#     pickle.dump(data, file)


with open("all_days_data.pkl", "rb") as file:
    loaded_data = pickle.load(file)


for i, day in enumerate(loaded_data.data):

    delta_t = i + 1

    X = []
    Y = []

    for j in range(day.shape[0] - delta_t):

        optical_flow = calculate_optical_flow(day['Image'][i])

        X.append(optical_flow)
        Y.append(day['SolarPower'][j+delta_t])

    model = model(train_model(X, Y))

    with open(f"model_for_day_{i}.pkl", "wb") as file:
        pickle.dump(model, file)
