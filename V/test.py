# pip install pandas numpy librosa scikit-learn

import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class AudioClassifier:
    """
    Class for audio classification using MFCC features and a neural network classifier.
    """

    def __init__(self):
        pass

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
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
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
        print("Validation Accuracy:", accuracy)

def main():
    """
    Main function to run audio classification.
    """
    csv_path = 'AUDIO FILES/train/train.csv'
    audio_dir = 'AUDIO FILES/train/Train'

    classifier = AudioClassifier()
    X, y = classifier.load_data(csv_path, audio_dir)
    model, X_val, y_val = classifier.train(X, y)
    classifier.evaluate(model, X_val, y_val)

if __name__ == "__main__":
    main()