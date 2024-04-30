import pandas as pd
from scipy.optimize import minimize
from PIL import Image
import numpy as np
import os
import re
import pickle
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2


def get_training_data(data, delta):
    X = []
    Y = []

    for day in data:

        for j in range(day.shape[0] - 1 - delta):
            opt_img = day["flow_images"].iloc[j + 1].to_numpy()
            power_generation = day["SolarPower"].iloc[j + 1 + delta]

            X.append(opt_img)
            Y.append(power_generation)

    return X, Y


class model:
    def __init__(self, betas, loss):
        self.loss = loss
        self.betas = betas

    def predict(self, data):
        def f(x):
            return abs((x[:, :, 0] @ self.betas[0]).T @ (x[:, :, 1] @ self.betas[1]))

        return [f(image) for image in data]


class multiday_prediction_model:
    def __init__(self, models):
        self.models = models
        self.loss = sum([model.loss for model in self.models])

    def predict(self, X):
        return np.array([model.predict(X) for model in self.models]).T


class optical_flow_image:
    def __init__(self, numpy_array):
        self.numpy = numpy_array

    def to_numpy(self):
        return self.numpy


class daily_data_structure:
    def __init__(self, data):
        self.all_days = data
        self.num_days = len(data)

    def get_days(self, indices):
        return [self.all_days[idx] for idx in indices]


def load_object(path_to_file):
    with open(path_to_file, "rb") as file:
        obj = pickle.load(file)
    return obj


def save_object(obj, file_path):
    with open(f"{file_path}.pkl", "wb") as file:
        pickle.dump(obj, file)


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
    def train_model(X, Y):

        def loss_function(beta, X, Y, lambdas, Gammas):
            n = int(len(beta) / 2)

            beta1 = beta[:n]
            beta2 = beta[n:]

            # loss = sum([(Y[i] - (X[i][0, :, :]@beta1).T@(X[i][1, :, :]@beta2)).T@(Y[i] - (X[i][0, :, :]@beta1).T@(X[i][1, :, :]@beta2)) for i in range(len(X))]) + lambdas[0]*beta1.T@Gammas[0]@beta1 + lambdas[1]*beta2.T@Gammas[1]@beta2

            guess = lambda i: abs((X[i][:, :, 0] @ beta1).T @ (X[i][:, :, 1] @ beta2))
            correction_term_1 = lambdas[0] * beta1.T @ Gammas[0] @ beta1
            correction_term_2 = lambdas[1] * beta2.T @ Gammas[1] @ beta2
            error_sum = sum([abs(Y[i] - guess(i)) for i in range(len(X))])/len(X)

            loss = error_sum #+ correction_term_1 + correction_term_2

            return loss

        n = X[0].shape[1]

        lambdas = [0.5, 0.5]
        Gammas = [np.eye(n), np.eye(n)]

        initial_guess = np.ones(2 * n)

        print("Training model...")

        result = minimize(loss_function,
                          initial_guess,
                          args=(X, Y, lambdas, Gammas),
                          method='SLSQP')

        print("Model trained!")

        beta = result.x
        optimal_loss = result.fun

        beta1 = beta[:n]
        beta2 = beta[n:]

        M = model((beta1, beta2), optimal_loss)

        return M, optimal_loss

    models = []

    for i in range(number_of_timesteps_to_predict):
        delta = i + 1

        X, Y = get_training_data(data, delta)

        trained_model, _ = train_model(X, Y)

        models.append(trained_model)


    multiday_predictor = multiday_prediction_model(models)

    if save_to_file:
        try:
            with open(f"{model_name}.pkl", "wb") as file:
                pickle.dump(multiday_predictor, file)
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

                grayscale_image = grayscale_squeezed
                # Convert the grayscale array back to an image
                # grayscale_image = Image.fromarray(grayscale_squeezed, 'L')

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
        all_data.append(
            pd.merge_asof(satellite_images.sort_index(), solar_power_generation_data.sort_index(), left_index=True,
                          right_index=True, direction='backward'))

    data = daily_data_structure(all_data)

    save_object(data, file_name)


# def transform_data_to_optical_flow(data):
#     class optical_flow_image:
#         def __init__(self, numpy_array):
#             self.numpy = numpy_array
#
#         def to_numpy(self):
#             return self.numpy
#
#     data_optical_flow = []
#
#     for i, day in enumerate(data):
#
#         data_optical_flow.append(day.copy())
#
#         data_optical_flow[i].loc[data_optical_flow[i].index[0], "Image"] = None
#         data_optical_flow[i].loc[data_optical_flow[i].index[-1], "Image"] = None
#
#         for j in range(day.shape[0] - 2):
#             optical_flow = calculate_optical_flow(np.array(day["Image"].iloc[j]), np.array(day["Image"].iloc[j + 1]))
#             data_optical_flow[i].loc[data_optical_flow[i].index[j + 1], "Image"] = optical_flow_image(optical_flow)
#
#     return data_optical_flow


def apply_denmark_mask(image, mask):
    """Applies the Denmark mask to an image."""
    return image * mask


def transform_data_to_optical_flow(data):

    data_optical_flow = []

    mask = np.load("data/mask.npy")

    for i, day in enumerate(data):

        day_copy = day.copy()

        # Reset the previous image for each day, account for jump between days
        prev_image = None
        flow_images = []

        for image in day_copy["Image"].tolist():

            current_image = apply_denmark_mask(image, mask)

            if prev_image is not None:

                flow = calculate_optical_flow(prev_image, current_image)
                flow_images.append(optical_flow_image(flow))

            else:
                flow_images.append(None)

            prev_image = current_image

        day_copy['Image'] = flow_images
        day_copy.rename(columns={'Image': 'flow_images'}, inplace=True)
        data_optical_flow.append(day_copy)

    return data_optical_flow


def calculate_optical_flow(prev_image, current_image):
    """ Calculates and returns the optical flow between two images. """
    return cv2.calcOpticalFlowFarneback(prev_image, current_image, None, 0.5, 3, 5, 3, 5, 1.2, 0)


def display_optical_flow_image(image: optical_flow_image):

    optical_flow = image.to_numpy()

    # Extract the X and Y components of the flow
    U = optical_flow[:, :, 0]  # X components
    V = optical_flow[:, :, 1]  # Y components

    # Create a grid of coordinates for the vectors
    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))

    # Plot the vector field using quiver
    plt.figure(figsize=(5, 5))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.axis('equal')  # Keep the scale of x and y equal
    plt.title('Optical Flow Field')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

def display_optical_flow_image_as_RGB(image: optical_flow_image):
    def min_max_scale(arr):
        # Compute the minimum and maximum values of the array
        min_val = np.min(arr)
        max_val = np.max(arr)

        # Apply the Min-Max scaling formula
        scaled_arr = (arr - min_val) / (max_val - min_val) * 255

        # Convert to integer type suitable for image data
        scaled_arr = scaled_arr.astype(np.uint8)

        return scaled_arr

    img = image.to_numpy()

    matrix_red = min_max_scale(img[:, :, 0])
    matrix_gb = min_max_scale(img[:, :, 1])

    # Create an empty RGB image with the same dimensions
    rgb_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Assign the red channel
    rgb_image[:, :, 0] = matrix_red

    # Assign the green and blue channels
    rgb_image[:, :, 1] = matrix_gb
    rgb_image[:, :, 2] = matrix_gb

    # Display the RGB image
    plt.imshow(rgb_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def data_to_input_format(data):
    X = []
    for day in data:

        for j in range(day.shape[0] - 1):
            optical_flow_image = day["flow_images"].iloc[j + 1].to_numpy()
            X.append(optical_flow_image)

    return X

def get_evaluation_data(df, delta):
    images = df['flow_images'].tolist()  # Convert the Image column to a list
    power = df['SolarPower'].tolist()  # Convert the Power column to a list

    power_slices = []
    for i in range(len(images)):
        # Slice the power list from the next index to index + delta
        power_slice = power[i + 1:i + delta + 1]
        power_slices.append(power_slice)

        # Print for checking - testing purposes
        # print(f"Image {i}: {type(images[i])}")
        # print(f"Solar Power for Image {i}: {power_slice}")
        # print(f"Original solar power for image {i}: {power[i+1:i+delta+1]}\n")

    return images, power_slices
    # Example usage: images, power = get_evaluation_data(training_data[0], 2)


def __main__():

    #preprocess_and_save_data("all_days_data")

    data = load_object("all_days_data.pkl")

    optical_flow_data = transform_data_to_optical_flow(data.all_days)

    optical_flow_data = daily_data_structure(optical_flow_data)

    training_data = optical_flow_data.get_days([0, 1, 2, 3, 4])
    test_data = optical_flow_data.get_days([5])

    get_model(data=training_data,
              number_of_timesteps_to_predict=5,
              save_to_file=True,
              model_name="five_time_step_model")

    model = load_object("five_time_step_model.pkl")

    dayid = 0


    display_optical_flow_image(test_data[dayid]["flow_images"].iloc[1])

    X = data_to_input_format([test_data[dayid].head(2)])

    predictions = model.predict(X)

    print(predictions[0])
    print("\n------------------\n")
    print(test_data[dayid]["SolarPower"].iloc[2:2+5].tolist())
    print("Model loss:", model.models[0].loss)


if __name__ == "__main__":
    __main__()
