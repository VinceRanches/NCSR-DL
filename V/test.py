# pip install pandas numpy librosa scikit-learn matplotlib seaborn

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Paths for training and test data
train_csv_path = 'AUDIO FILES\\train\\train.csv'
train_audio_dir = 'AUDIO FILES\\train\\Train'
test_csv_path = 'AUDIO FILES\\test\\test.csv'
test_audio_dir = 'AUDIO FILES\\test\\Test'
output_csv = 'AUDIO FILES\\test\\test_predictions.csv'

class AudioClassifier:
    """
    Class for audio classification using MFCC features and a neural network classifier.
    """        
    def __init__(self):
        pass

    def explore_data(self, csv_path, audio_dir, num_samples=5):
        """
        Performs exploratory analysis on the audio data.

        Args:
            csv_path (str): Path to the CSV file.
            audio_dir (str): Directory containing audio files.
            num_samples (int): Number of samples to visualize per class.
        """
        # Load the data
        train_data = pd.read_csv(csv_path)

        # Display basic information about the dataset
        print("Dataset information:")
        print(train_data.info())
        
        # Display class distribution
        print("\nClass distribution:")
        class_distribution = train_data['Class'].value_counts()
        print(class_distribution)
        
        # Generate colors from the 'jet' colormap
        color_palette = plt.cm.jet(np.linspace(0, 1, len(class_distribution)))

        # Plot class distribution
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(2, 1, 1)
        class_distribution.plot(kind='bar', color=color_palette, ax=ax1)
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        for i, count in enumerate(class_distribution):
            ax1.text(i, count + 10, str(count), ha='center')
        manager = plt.get_current_fig_manager()
        manager.window.geometry("+{}+{}".format(0, 0))  # Spawn to top left corner
        plt.pause(0.1)  # Pause to allow the window to be positioned correctly
        plt.show()
        
        # Duration distribution of audio files
        durations = []
        for idx, row in train_data.iterrows():
            file_path = os.path.join(audio_dir, str(row['ID']) + '.wav')
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(duration)

        plt.figure(figsize=(10, 7))
        sns.histplot(durations, bins=30)
        plt.title('Duration Distribution of Audio Files')
        plt.xlabel('Duration (s)')
        plt.ylabel('Count')
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.window.geometry("+{}+{}".format(0, 0))  # Spawn to top left corner
        plt.pause(0.1)
        plt.show()

        # Visualize MFCC features for a few samples from each class
        classes = train_data['Class'].unique()
        plt.figure(figsize=(16, 8))
        for i, cls in enumerate(classes):
            plt.subplot(3, 4, i + 1)
            plt.title(cls)
            class_samples = train_data[train_data['Class'] == cls].sample(num_samples)
            for _, row in class_samples.iterrows():
                file_path = os.path.join(audio_dir, str(row['ID']) + '.wav')
                y, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                librosa.display.specshow(mfcc, sr=sr, x_axis='time')
                plt.ylabel('MFCC Coefficients')
                plt.xlabel('Time')
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.window.geometry("+{}+{}".format(0, 0))  # Spawn to top left corner
        plt.pause(0.1)
        plt.show()
        
    def extract_features(self, file_path):
        """
        Extracts MFCC features along with zero crossing rate and spectral centroids from an audio file.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            np.ndarray: Concatenated feature vector.
        """
        y, sr = librosa.load(file_path, sr=None)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=20), axis=1) # n_mfcc within [13, 20] and n_mels within [20, 40]
        return np.concatenate((mfcc, [zero_crossing_rate, spectral_centroids]))

    def load_data(self, csv_path, audio_dir):
        """
        Loads audio data from CSV file containing file IDs and classes.

        Args:
            csv_path (str): Path to the CSV file.
            audio_dir (str): Directory containing audio files.

        Returns:
            tuple: Feature vectors (X) and corresponding labels (y).
        """
        train_data = pd.read_csv(csv_path)
        X, y = [], []
        for idx, row in train_data.iterrows():
            file_path = os.path.join(audio_dir, str(row['ID']) + '.wav')
            features = self.extract_features(file_path)
            X.append(features)
            y.append(row['Class'])
        return np.array(X), np.array(y)

    def train(self, X, y):
        """
        Trains the neural network classifier.

        Args:
            X (np.ndarray): Feature vectors.
            y (np.ndarray): Labels.

        Returns:
            tuple: Trained model, validation data and labels.
        """
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, learning_rate='adaptive', random_state=42)
        model.fit(X_train, y_train)
        return model, X_val, y_val

    def evaluate(self, model, X_val, y_val):
        """
        Evaluates the trained model on validation data.

        Args:
            model: Trained model.
            X_val (np.ndarray): Validation feature vectors.
            y_val (np.ndarray): Validation labels.
        """
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print("(Train) Accuracy:", accuracy)

    def load_test_data(self, csv_path, audio_dir):
        """
        Loads test audio data from CSV file containing file IDs.

        Args:
            csv_path (str): Path to the CSV file.
            audio_dir (str): Directory containing test audio files.

        Returns:
            np.ndarray: Feature vectors.
            list: File IDs.
        """
        test_data = pd.read_csv(csv_path)
        X_test, file_ids = [], []
        for idx, row in test_data.iterrows():
            file_path = os.path.join(audio_dir, str(row['ID']) + '.wav')
            features = self.extract_features(file_path)
            X_test.append(features)
            file_ids.append(row['ID'])
        return np.array(X_test), file_ids

    def predict_test_data(self, model, X_test):
        """
        Predicts labels for the test data using the trained model.

        Args:
            model: Trained model.
            X_test (np.ndarray): Test feature vectors.

        Returns:
            np.ndarray: Predicted labels.
        """
        return model.predict(X_test)

    def save_predictions(self, file_ids, predictions, output_csv):
        """
        Saves predictions to a CSV file.

        Args:
            file_ids (list): File IDs.
            predictions (np.ndarray): Predicted labels.
            output_csv (str): Path to the output CSV file.
        """
        df = pd.DataFrame({'ID': file_ids, 'Class': predictions})
        df.to_csv(output_csv, index=False)


def main():
    """
    Main function to finalize and validate the audio classification model on the test set.
    """

    # Instantiate the classifier
    classifier = AudioClassifier()

    # Explore the data
    classifier.explore_data(train_csv_path, train_audio_dir)

    # # Load and preprocess training data
    # X_train, y_train = classifier.load_data(train_csv_path, train_audio_dir)

    # # Train the model
    # model, X_val, y_val = classifier.train(X_train, y_train)

    # # Evaluate the model on validation data
    # classifier.evaluate(model, X_val, y_val)

    # # Load and preprocess test data
    # X_test, file_ids = classifier.load_test_data(test_csv_path, test_audio_dir)
    # scaler = StandardScaler()
    # X_test_scaled = scaler.fit_transform(X_test)

    # # Predict labels for test data
    # predictions = classifier.predict_test_data(model, X_test_scaled)

    # # Save predictions to CSV
    # classifier.save_predictions(file_ids, predictions, output_csv)

if __name__ == "__main__":
    main()
