import pandas as pd
from scipy.optimize import minimize
import numpy as np
import os
import re
import pickle
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys


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


class model:
    def __init__(self, betas, loss):
        self.loss = loss
        self.betas = betas

    def predict(self, image, current_power_generation):

        def f(X):
            return np.sqrt((current_power_generation + ((X[:, :, 0] @ self.betas[0])+(X[:, :, 1] @ self.betas[1])).T @ (X[:, :, 2] @ self.betas[2]))**2)
        return f(image)


class multiday_prediction_model:
    def __init__(self, models, delta):
        self.delta = delta
        self.models = models
        self.loss = sum([model.loss for model in self.models])

    def predict(self, df):

        X = data_to_input_format([df])
        current_power_generation = [x[0] for x in df['SolarPower'].tolist()]
        predictions = []

        for i, image in enumerate(X):
            start_date = df.index[i]
            prediction_timestamps = generate_dates(start_date + pd.Timedelta(minutes=1), self.delta)
            prediction = []
            for model in self.models:
                prediction.append(model.predict(image, current_power_generation[i]))
            prediction=pd.DataFrame(prediction, index=prediction_timestamps)
            predictions.append(prediction)

        return predictions

    def evaluate(self, df: pd.DataFrame, error_measure):

        _, Y = get_evaluation_data(df, self.delta)

        xs = []
        ys = []

        predictions = self.predict(df)

        for i, prediction in enumerate(predictions):
            xs += prediction.tolist()
            ys += Y[i]

        error = error_measure(xs, ys)

        return error


def get_training_data(data, delta):

    X = []
    Y = []

    for day in data:

        try:
            for j in range(day.shape[0] - delta):
                opt_img = day["flow_images"].iloc[j].to_numpy()
                idx = day.index[j]
                power_generation = get_nth_power_after_index(day, n=delta, index=idx, power_column_name="SolarPower")

                X.append(opt_img)
                Y.append(power_generation)
        except:
            pass  # just stop when there is no more data left

    return X, Y


def generate_dates(start_date, n):
    dates = [start_date]
    for i in range(1, n):
        next_date = start_date + pd.Timedelta(minutes=i)
        dates.append(next_date)
    return dates


def RMSE(xs, ys):
    return np.sqrt(sum([(x-y)**2/len(xs) for x, y in zip(xs,ys)]))


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


def get_model(data, number_of_timesteps_to_predict, save_to_file=False, model_name="multiday_model", verbose=False):
    def train_model(current_power_generation, X, Y):

        def loss_function(beta, current_power_generation, X, Y, lambdas, Gammas):
            n = len(beta) // 3
            beta1 = beta[:n]
            beta2 = beta[n:2 * n]
            beta3 = beta[2 * n:]

            guess = lambda i: np.sqrt((current_power_generation[i] + ((X[i][:, :, 0] @ beta1)+(X[i][:, :, 1] @ beta2)).T @ (X[i][:, :, 2] @ beta3))**2)
            correction_term_1 = lambdas[0] * beta1.T @ Gammas[0] @ beta1
            correction_term_2 = lambdas[1] * beta2.T @ Gammas[1] @ beta2
            correction_term_3 = lambdas[2] * beta3.T @ Gammas[2] @ beta3
            error_sum = np.sqrt(sum([(Y[i] - guess(i))**2 for i in range(len(X))])/len(X))

            loss = error_sum + correction_term_1 + correction_term_2 + correction_term_3

            return loss

        # Function to track loss during optimization
        loss_evolution = []

        def callback_loss(x):
            loss_evolution.append(loss_function(x, current_power_generation, X, Y, lambdas, Gammas))

        n = X[0].shape[1]

        gamma = 10**4
        lambdas = [gamma/3 for _ in range(3)]
        Gammas = [np.eye(n), np.eye(n), np.eye(n)]

        initial_guess = np.ones(3 * n)*0.00001

        #print("Training model...")

        result = minimize(loss_function,
                          initial_guess,
                          args=(current_power_generation, X, Y, lambdas, Gammas),
                          method='SLSQP', #'L-BFGS-B', #'SLSQP',
                          #tol=1e-6,  # Adjust tolerance
                          options={'maxiter': 200},
                          callback=callback_loss)

        #print("Model trained!")

        # Plot loss evolution
        plt.plot(loss_evolution)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Loss Evolution during Optimization')
        plt.show()

        beta = result.x
        optimal_loss = result.fun

        print("Loss:", optimal_loss)

        n = len(beta) // 3

        beta1 = beta[:n]
        beta2 = beta[n:2 * n]
        beta3 = beta[2 * n:]

        M = model((beta1, beta2, beta3), optimal_loss)

        return M, optimal_loss

    models = []

    if verbose:
        print(f"\n(0/{number_of_timesteps_to_predict}) Training model...")

    for i in range(number_of_timesteps_to_predict):
        delta = i + 1

        X, Y = get_training_data(data, delta)
        current_power_generation = []
        for day in data:
            current_power_generation += [x[0] for x in day['SolarPower'].tolist()]

        trained_model, _ = train_model(current_power_generation, X, Y)

        if verbose:
            print(f"({i+1}/{number_of_timesteps_to_predict}) Training model...")

        models.append(trained_model)


    multiday_predictor = multiday_prediction_model(models=models, delta=number_of_timesteps_to_predict)

    if save_to_file:
        try:
            with open(f"{model_name}.pkl", "wb") as file:
                pickle.dump(multiday_predictor, file)
        except:
            print(f"Unable to save model to file: {model_name}.pkl!")

    return multiday_predictor


def pair_images_with_power_data(df: pd.DataFrame,
                                image_column_name: str,
                                power_column_name: str):

    # Get the indices of non-NaN elements in the specified column
    indices = df.index[df[image_column_name].notna()]
    pairs = []
    for i, idx in enumerate(indices):
        try:
            pairs.append((df[power_column_name].loc[idx:indices[i+1]].tolist(), df[image_column_name].loc[idx]))
        except:
            pairs.append((df[power_column_name].loc[idx:].tolist(), df[image_column_name].loc[idx]))

    df = pd.DataFrame(pairs, columns=df.columns)
    df.index = indices
    return df

def get_nth_power_after_index(df, n, index, power_column_name):
    element = None
    k = 1

    if isinstance(index, int):
        while True:
            try:
                element = df.iloc[index + k-1][power_column_name][n - sum([len(df.iloc[index + i][power_column_name]) for i in range(k-1)])]
                break
            except IndexError:
                if k > df.iloc[index:].shape[0]:
                    print("Index out of bounds!")
                    break
            k += 1
    elif isinstance(index, pd.Timestamp):
        while True:
            try:
                target_date = index + pd.Timedelta(days=k-1)
                element = df.loc[target_date, power_column_name][n - sum([len(df.loc[target_date, power_column_name]) for i in range(k-1)])]
                break
            except KeyError:
                if k > df.iloc[index:].shape[0]:
                    print("Index out of bounds!")
                    break
            k += 1
    else:
        raise ValueError("Unsupported index type. Only integer and datetime indices are supported.")

    return element


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
    all_merged_data = []

    for filename in os.listdir('data/csv_files'):
        satellite_images = load_satellite_images(f'data/{os.path.splitext(filename)[0]}')

        # Round up the minutes
        satellite_images.index = satellite_images.index.ceil('min')

        solar_power_generation_data = load_solar_power_generation_data(f'data/csv_files/{filename}')

        # Perform left join
        merged_data = pd.merge(solar_power_generation_data, satellite_images, left_index=True, right_index=True,
                               how='left')

        # all_data.append(
        #     pd.merge_asof(satellite_images.sort_index(), solar_power_generation_data.sort_index(), left_index=True,
        #                   right_index=True, direction='backward'))

        merged_data = merged_data.sort_values(by='Minutes1UTC', ascending=True)

        all_merged_data.append(merged_data)


        all_data.append(pair_images_with_power_data(merged_data,
                                        image_column_name='Image',
                                        power_column_name='SolarPower'))


    merged_data = daily_data_structure(all_merged_data)
    data = daily_data_structure(all_data)

    save_object(merged_data, "merged_data")
    save_object(data, file_name)


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

                # Include current sattelite image
                flow = np.concatenate((flow, current_image[:, :, np.newaxis]), axis=2)

                flow_images.append(optical_flow_image(flow))

            else:
                flow_images.append(None)

            prev_image = current_image

        day_copy['Image'] = flow_images
        day_copy.rename(columns={'Image': 'flow_images'}, inplace=True)
        data_optical_flow.append(day_copy.dropna())

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

        for j in range(day.shape[0]):
            optical_flow_image = day["flow_images"].iloc[j].to_numpy()
            X.append(optical_flow_image)

    return X


def get_evaluation_data(df, delta):

    X = []
    Y = []

    try:
        for j in range(df.shape[0] - delta):
            opt_img = df["flow_images"].iloc[j].to_numpy()
            idx = df.index[j + 1]
            y = []
            for d in range(delta):
                time_steps = d + 1
                power_generation = get_nth_power_after_index(df, n=time_steps, index=idx, power_column_name="SolarPower")
                y.append(power_generation)
            X.append(opt_img)
            Y.append(y)

    except:
        pass  # just stop when there is no more data left


    return X, Y
    # Example usage: images, power = get_evaluation_data(training_data[0], 2)


def split_dataframe(df, n):
    chunk_size = len(df) // n
    chunks = []
    for i in range(n):
        start = i * chunk_size
        if i == n - 1:
            chunks.append(df[start:])
        else:
            chunks.append(df[start:start + chunk_size])
    return chunks


def cross_validate(get_model, delta, dfs: list, n_splits=1):

    dfs_split = []

    for df in dfs:
        dfs_split += split_dataframe(df, n_splits)

    scores = []

    print("Calculating cross validation score...")

    for i in tqdm(range(len(dfs_split))):
        training_data = dfs_split.copy()
        training_data.pop(i)
        test_data = dfs_split[i]
        model = get_model(data=training_data,
                          number_of_timesteps_to_predict=delta,
                          save_to_file=True,
                          model_name=f"{delta}_time_step_model({i})")

        print("Model #", i)
        print("Model loss:", model.loss)

        score = model.evaluate(test_data, error_measure=RMSE)
        scores.append(score)

    print("Cross validation scores:", scores)

    cross_validation_score = np.mean(scores)

    return cross_validation_score


def evaluate_overfitting(get_model, delta, dfs: list, n_splits=1):

    dfs_split = []

    for df in dfs:
        dfs_split += split_dataframe(df, n_splits)

    scores = []

    print("Calculating overfitting score...")

    for i in tqdm(range(len(dfs_split))):
        training_data = dfs_split.copy()
        training_data.pop(i)
        model = get_model(data=training_data,
                          number_of_timesteps_to_predict=delta,
                          save_to_file=False,
                          model_name=f"{delta}_time_step_model{i}")
        print("Model #", i)
        print("Model loss:", model.loss)

        for df in training_data:
            score = model.evaluate(df, error_measure=RMSE)
            scores.append(score)

    cross_validation_score = cross_validate(get_model, delta=delta, dfs=dfs, n_splits=n_splits)

    print("Cross validation score:", cross_validation_score)

    overfitting_score =  (cross_validation_score - np.mean(scores)) / cross_validation_score

    return overfitting_score


def display_predictions(model, df, merged_data, historical_points=10):

    predictions = model.predict(df)

    for i in range(df.shape[0]):

        optical_flow = df["flow_images"].iloc[i].to_numpy()

        start_date = df.index[i] - pd.Timedelta(minutes=historical_points)
        end_date = df.index[i] + pd.Timedelta(minutes=historical_points)

        truth = merged_data["SolarPower"].loc[start_date:end_date]

        # Extract the X and Y components of the flow
        U = optical_flow[:, :, 0]  # X components
        V = optical_flow[:, :, 1]  # Y components
        magnitude = np.sqrt(U ** 2 + V ** 2)

        # Create a grid of coordinates for the vectors
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))

        # Plot the vector field using quiver
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Create a subplot with 1 row and 2 columns

        ax1.imshow(optical_flow[:, :, 2], cmap='gray')
        ax1.set_title("Current sky state")

        # Plot the quiver plot on the same axes
        ax1.quiver(X, Y, U, V,
                  magnitude,
                  angles='xy',
                  scale_units='xy',
                  scale=1,
                  cmap='plasma')
        ax1.axis('equal')  # Keep the scale of x and y equal
        ax1.set_title('Optical Flow Field')

        # Plot the data in the subplot
        ax2.plot(predictions[i], marker='.', color='red', label="Prediction")
        ax2.plot(truth, marker='.', color='blue', label="True data")
        ax2.axvline(x=df.index[i], color='black', linestyle='--')
        ax2.set_title(f'Start time: {df.index[i]}')
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel('t')
        ax2.set_ylabel('Solar Power (MW)')

        plt.show()


def __main__():

    original_stdout = sys.stdout

    with open("output_15_minute_model.log", 'w') as file:
        sys.stdout = file

        preprocess_and_save_data("15_minutes_model_data")

        data = load_object("15_minutes_model_data.pkl")

        optical_flow_data = transform_data_to_optical_flow(data.all_days)

        optical_flow_data = daily_data_structure(optical_flow_data)

        overfitting_score = evaluate_overfitting(get_model, delta=15, dfs=optical_flow_data.all_days, n_splits=1)

        print("Overfitting score:", overfitting_score)

    sys.stdout = original_stdout


if __name__ == "__main__":
    __main__()
