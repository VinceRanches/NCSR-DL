import tensorflow as tf
from tensorflow.keras import backend as K
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def f1_score(y_true, y_pred):

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1 = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))
    return f1


def create_model_NN(input_shape = (138, ), output = 10):


    model = Sequential([
    layers.Dense(1024, activation='relu', input_shape = input_shape),
    layers.BatchNormalization(axis=-1),
    layers.Dropout(0.2),
    
    layers.Dense(110, activation='relu'),
    layers.BatchNormalization(axis=-1),
    layers.Dropout(0.2),
    
    layers.Dense(110, activation='relu'),
    layers.BatchNormalization(axis=-1),
    layers.Dropout(0.2),

    layers.Dense(310, activation='relu'),
    layers.BatchNormalization(axis=-1),
    layers.Dropout(0.2),
    
    layers.Dense(710, activation='relu'),
    layers.BatchNormalization(axis=-1),
    layers.Dropout(0.2),

    layers.Dense(310, activation='relu'),
    layers.BatchNormalization(axis=-1),
    layers.Dropout(0.2),
    
    layers.Dense(output, activation='softmax')
    ])

    return model


def create_model_CNN(input_shape=(128, 100, 1), num_classes=10):
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    return model


def create_model_CNN_AS(input_shape=(128, 100, 1), num_classes=8):
    """
    Creates a Convolutional Neural Network (CNN) model for classification.

    Parameters:
    - input_shape (tuple): Shape of the input data (height, width, channels).
    - num_classes (int): Number of classes for classification.

    Returns:
    - model (Sequential): Compiled Keras Sequential model.
    """
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    # Flatten layer
    model.add(Flatten())

    # Dense layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    return model



def plot_training_history(training_score, val_score, graph_name):
    """
    Plots the training and validation f1 scores over epochs.

    Parameters:
    training_score (list): List of trainingscores for each epoch.
    val_score (list): List of validation cores for each epoch.
    graph_name (str): Name of the graph.
    """
    epochs = range(1, len(training_score) + 1)

    plt.plot(epochs, training_score, '-', label='Training')
    plt.plot(epochs, val_score, ':', label='Validation')
    plt.title(graph_name)
    plt.xlabel('Epoch')
    plt.ylabel('')
    plt.legend(loc='lower right')
    plt.show()



def plot_confusion_matrix(model, x_val, y_val, classes):
    """
    Evaluates the model on validation data and plots the confusion matrix.

    Parameters:
    - model: The trained model to be evaluated.
    - x_val: Validation features.
    - y_val: True labels for validation data.
    - classes: List of class labels.

    Returns:
    - accuracy: The accuracy of the model on the validation data.
    """

    # Calculate predicted labels
    y_predicted = model.predict(x_val)

    # Convert one-hot encoded labels back to categorical labels
    y_val_categorical = [classes[idx] for idx in y_val.argmax(axis=1)]
    y_pred_categorical = [classes[idx] for idx in y_predicted.argmax(axis=1)]

    # Calculate accuracy
    accuracy = accuracy_score(y_val_categorical, y_pred_categorical)
    print("Accuracy:", accuracy)

    # Calculate confusion matrix
    conf_mat = confusion_matrix(y_val_categorical, y_pred_categorical)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title('Confusion Matrix')
    plt.show()

def model_load(model_path):
    model = load_model(model_path, custom_objects={'f1_score': f1_score})
      # Load the model
    print(model.summary())
    
    return model 

def freeze_and_delete_layers(model, num_layers_to_freeze):
    # Freeze the first num_layers_to_freeze layers
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    
    # Pop all layers beyond the first num_layers_to_freeze
    while len(model.layers) > num_layers_to_freeze:
        model.pop()

    


