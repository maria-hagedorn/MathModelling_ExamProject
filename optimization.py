import pandas as pd
from scipy.optimize import minimize
from PIL import Image
import numpy as np
import os
import re
import pickle
import tifffile as tiff
import matplotlib.pyplot as plt


class model:
    def __init__(self, predictor, loss):
        self.loss = loss
        self.predict = predictor


class daily_data_structure:
    def __init__(self, data):
        self.all_days = data
        self.num_days = len(data)

    def get_day(self, index):
        return self.data[index-1]



def extract_date_time_from_filenames(folder_path):

    pattern = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")
    date_time_tuples = []

    for filename in os.listdir(folder_path):
        match = pattern.search(filename)

        if match:
            year, month, day, hour, minute, second = match.groups()
            date_time_tuples.append((f"{year}-{month}-{day}", f"{hour}:{minute}:{second}"))

    return date_time_tuples


def get_model(data, number_of_timesteps_to_predict, save_to_file=False, model_name="multiday_model"):
    def get_training_data(data, delta):

        X = []
        Y = []

        for day in data:

            for j in range(day.shape[0] - 1 - delta):
                optical_flow_image = day["Image"][j + 1]
                power_generation = day["SolarPower"][j + 1 + delta]

                X.append(optical_flow_image)
                Y.append(power_generation)

        return X, Y

    def train_model(X, Y):

        def loss_function(beta, X, Y, lambdas, Gammas):
            n = int(len(beta) / 2)

            beta1 = beta[:n]
            beta2 = beta[n:]

            # loss = sum([(Y[i] - (X[i][0, :, :]@beta1).T@(X[i][1, :, :]@beta2)).T@(Y[i] - (X[i][0, :, :]@beta1).T@(X[i][1, :, :]@beta2)) for i in range(len(X))]) + lambdas[0]*beta1.T@Gammas[0]@beta1 + lambdas[1]*beta2.T@Gammas[1]@beta2
            loss = sum([Y[i] - (X[i][0, :, :] @ beta1).T @ (X[i][1, :, :] @ beta2) for i in range(len(X))]) + lambdas[
                0] * beta1.T @ Gammas[0] @ beta1 + lambdas[1] * beta2.T @ Gammas[1] @ beta2

            return loss

        n = X[0].shape[2]

        lambdas = [0.5, 0.5]
        Gammas = [np.eye(n), np.eye(n)]

        initial_guess = np.ones(2 * n)

        result = minimize(loss_function,
                          initial_guess,
                          args=(X, Y, lambdas, Gammas),
                          method='SlSQP')

        beta = result.x
        optimal_loss = result.fun

        beta1 = beta[:n]
        beta2 = beta[n:]

        predicter = lambda x: (x[0, :, :] @ beta1).T @ (x[1, :, :] @ beta2)

        M = model(predicter, optimal_loss)

        return M, optimal_loss

    models = []

    for i in range(number_of_timesteps_to_predict):

        delta = i + 1

        X, Y = get_training_data(data, delta)

        models.append(model(train_model(X, Y)))

    class multiday_prediction_model:
        def __init__(self, models):
            self.models = models

        def predict(self, X):
            return [model.predict(X) for model in self.models]


    multiday_predictor = multiday_prediction_model(models)

    if save_to_file:
        try:
            with open(f"{model_name}.pkl", "wb") as file:
                pickle.dump(model, file)
        except:
            print(f"Unable to save model to file: {model_name}.pkl!")

    return multiday_predictor


def preprocess_and_save_data(file_name="all_days_data"):
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

                grayscale = R + G - 2 * B

                # Squeeze the grayscale values to the 0-255 range
                min_val = np.min(grayscale)
                max_val = np.max(grayscale)
                if max_val != min_val:
                    grayscale_squeezed = 255 * (grayscale - min_val) / (max_val - min_val)
                else:
                    grayscale_squeezed = np.full(grayscale.shape,
                                                 fill_value=(1 - np.round(1 / (min_val + 0.0001)) * min(min_val, 0)) * (
                                                     min(min_val, 255)), dtype=np.uint8)

                # Convert the grayscale array back to an image
                grayscale_image = Image.fromarray(grayscale_squeezed, 'L')

                images_with_date_times.append((grayscale_image, date_time))

        images = pd.DataFrame({
            'Image': [tup[0] for tup in images_with_date_times],
            'DateTime': [' '.join(tup[1]) for tup in images_with_date_times]
            # Combining date and time into a single string
        })
        images['DateTime'] = pd.to_datetime(images['DateTime'])
        images.set_index('DateTime', inplace=True)

        return images

    def load_solar_power_generation_data(file_path):
        excel_data = pd.read_csv(file_path, sep=",", parse_dates=['Minutes1UTC'], index_col='Minutes1UTC')
        excel_data = excel_data.drop(columns=['Minutes1DK'])
        solar_power_generation_data = excel_data['SolarPower']
        return solar_power_generation_data

    # READ AND SAVE DATA
    all_data = []

    for filename in os.listdir('data/csv_files'):
        satellite_images = load_satellite_images(f'data/{os.path.splitext(filename)[0]}')
        solar_power_generation_data = load_solar_power_generation_data(f'data/csv_files/{filename}')
        all_data.append(pd.merge_asof(satellite_images.sort_index(), solar_power_generation_data.sort_index(), left_index=True, right_index=True, direction='backward'))

    data = daily_data_structure(all_data)

    # Serialize the data object to a file
    with open(f"{file_name}.pkl", "wb") as file:
        pickle.dump(data, file)


def load_processed_data(path_to_file):
    with open(f"{path_to_file}.pkl", "rb") as file:
        loaded_data = pickle.load(file)

    return loaded_data


def transform_data_to_optical_flow(data):

    data_optical_flow = []

    for i, day in enumerate(data):

        data_optical_flow.append(day.copy())

        data_optical_flow[i]["Image"][0] = None
        data_optical_flow[i]["Image"][-1] = None

        for j in range(day.shape[0]-2):

            optical_flow = calculate_optical_flow(day["Image"][j], day["Image"][j+1])
        data_optical_flow[i]["Image"][j+1] = optical_flow

    return data_optical_flow


def __main__():

    preprocess_and_save_data("all_days_data")

    data = load_processed_data("all_days_data.pkl")

    optical_flow_data = transform_data_to_optical_flow(data)

    optical_flow_data = daily_data_structure(optical_flow_data)

    model = get_model(data=optical_flow_data.all_days,
                      number_of_timesteps_to_predict=5,
                      save_to_file=True,
                      model_name="five_time_step_model")

    model.predict(X)






