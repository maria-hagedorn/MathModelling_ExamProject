import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd





exit()

# Create the DataFrame
data = {
    'A': [1, 2, 3, 1, 1, 1, 2, 3, 4, 4, 4, 1, 1, 0, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
    'B': [None, None, None, None, 'A', None, None, None, None, None, None, None, None, None, None, 'B', None, None, None, None, None, None, None, None, None, None, 'D', None, None, None]
}

# Define the start datetime and number of periods
start_datetime = '2024-01-01 00:00:00'
num_periods = len(data['A'])

# Create a DatetimeIndex with minute frequency
datetime_index = pd.date_range(start=start_datetime, periods=num_periods, freq='min')

# Create the DataFrame with datetime indices
df = pd.DataFrame(data, index=datetime_index)

print(df)


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



df = pair_images_with_power_data(df, 'B', 'A')
print(df)


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

idx = df.index[2]
print("INDEX:", idx)

e = get_nth_power_after_index(df, 1, index=idx, power_column_name='A')
print(e)

exit()











with open("all_days_data.pkl", "rb") as file:
    loaded_data = pickle.load(file)



with open("all_days_flow_data.pkl", "wb") as file:
    pickle.dump(data_optical_flow, file)
