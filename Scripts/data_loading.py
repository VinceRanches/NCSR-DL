import os
import numpy as np

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
encoder = LabelEncoder()




def datapreprocessing(x, y, dataset):
    le = LabelEncoder()
    x = np.array(x)

    y_encoded = le.fit_transform(y)
    y_encoded = to_categorical(y_encoded)
    # This is needed because of how we dowloaded the data
    if dataset == "YamNet":
      x = np.reshape(x, (len(x), -1))

    return x, y_encoded, le

def load_arrays_and_labels(directory):
    x = []
    y = []

    for class_name in os.listdir(directory):
        class_directory = os.path.join(directory, class_name)

        for file_name in os.listdir(class_directory):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_directory, file_name)

                # Load the numpy array
                array = np.load(file_path)

                # Append the array to x and the class name to y
                x.append(array)
                y.append(class_name)

    return x, y



def load_data(dataset_path, dataset):
    """
    Prepares the data for training and validation.

    Parameters:
    dataset (str): The name of the dataset, e.g., "audioset".
    dataset_path (str): The path to the dataset.

    Returns:
    x_train, x_val, y_train, y_val
    """
    if dataset == "audioset":
        x_train, y_train = load_arrays_and_labels(f'{dataset_path}/Train_Features')
        x_val, y_val = load_arrays_and_labels(f'{dataset_path}/Eval_Features')
        x_train, y_train,le = datapreprocessing(x_train, y_train,dataset)
        x_val, y_val, _ = datapreprocessing(x_val, y_val,dataset)
        x_train, y_train = shuffle(x_train, y_train)
        x_val, y_val = shuffle(x_val, y_val)
        
    elif dataset == "YamNet":

        x, y = load_arrays_and_labels(dataset_path)
        x, y, le = datapreprocessing(x, y, dataset)
        x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

    else:
        x, y = load_arrays_and_labels(dataset_path)
        x, y, le = datapreprocessing(x, y, dataset)
        x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)
    
    return x_train, x_val, y_train, y_val, le

