import librosa
import numpy as np
import pandas as pd
import shutil
import tempfile
import os
import librosa
import IPython.display as ipd
from IPython.display import clear_output
from Scripts import Utilities as ut
from Scripts import Feature_Extraction as fe
from Scripts import Model_Training as mt
from Scripts import data_loading as dl
from IPython.display import clear_output
from deep_audio_features.bin.config import WINDOW_LENGTH, HOP_LENGTH
from sklearn.preprocessing import LabelEncoder
from deep_audio_features.utils import sound_processing as sp
from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction as dW


"""
The next two functions is to geth one sample predicttions

"""

def get_prediction(file_path, model, model_type):

    """

    This is a function that takes us input the wav file path the model type and the specific pretrained model eit extracts the needed features ands makes predictions live.

    """
    Stop_Yamnet =  False
    x,fs = librosa.load(file_path,sr = 42000)

    classes = ["air_conditioner", "car_horn","children_playing", "dog_bark", "drilling","engine_idling", "gun_shot","jackhammer", "siren", "street_music" ]
    # Load the CSV file
    csv_path = 'UrbanSound\\UrbanSound8K\\metadata\\UrbanSound8K.csv'
    df = pd.read_csv(csv_path)
    
    # Extract the file name from the file path
    file_name = file_path.split('\\')[-1]
    
    # Check if the file_name exists in the 'slice_file_name' column
    if file_name in df['slice_file_name'].values:
        # Get the corresponding class
        file_class = df.loc[df['slice_file_name'] == file_name, 'class'].values[0]
        print('############################################################################################\n')
        print(f"The Ground Truth is {file_class}\n")
        print('############################################################################################')
    else:

        print('############################################################################################\n')
        print("File not found in the CSV\n")
        print('############################################################################################')
        """
        Because the model only takes inputs which we already extracted features because we dont have the p[ackages to run it localy 
        if we run it in colab we could do it.

         """
        Stop_Yamnet = True
        
    
    if model_type == 'CNN':

        # For CNN we have different feature extraction tha fully connected thats why we have this if statement

        
        win_length=int(WINDOW_LENGTH * fs),
        hop_length=int(HOP_LENGTH * fs)
        mels = librosa.feature.melspectrogram(y=x, sr=fs, hop_length=hop_length)
        mels = librosa.power_to_db(mels, ref=np.max)
        mels = np.array(mels)

        mels = mels[:, :81]
        if mels.shape[1] < 81:
            mels = np.pad(mels, ((0, 0), (0, 81 - mels.shape[1])), 'constant')

        test = []
        test.append(mels)
        mels = np.array(test)
        print(test[0].shape)
        

    elif model_type == 'YamNet':

        if Stop_Yamnet:
            
            print('############################################################################################\n')
            print("We can only run from test folder of UrbanSound8k for other sound please select another mode\n")
            print('############################################################################################')
            return

        else:
            # We have rhe file name now we match it with the numpy file
            new_file_name = os.path.splitext(file_name)[0] + ".npy"
            mels = np.load(os.path.join("UrbanSound\Features\YamNet_Features\Test_Features", file_class, new_file_name))
        
            
    else:

        # Create a  temporary folder because this is hopw the PyaudioAnal;ysis works
        with tempfile.TemporaryDirectory() as temp_dir:
                    
            temp_file_path = os.path.join(temp_dir, file_name)

            shutil.copy(file_path, temp_file_path)

            mels, _, fn = dW(temp_dir, 1, 1, 0.1, 0.1)

        mels = mels.reshape(1, 138)
    
    # Get the predicted probabilities for each class
    probabilities = model.predict(mels)
    
    # Threshold for comparison
    threshold = 0.02
    
    # Get the positions where values are greater than the threshold
    positions = np.where(probabilities > threshold)

    if len(positions[0]) == 1:
        x = positions[1][0]
        prob_x = probabilities[0][x] * 100
        formatted_prob_x = f"{prob_x:.2f}%"
        print('############################################################################################\n')
        print(f"The model believes this sound is {classes[x]} with  {formatted_prob_x} confidence,\n")
        print('############################################################################################')
    elif len(positions[0]) == 2:
        positions = sorted(positions, key=lambda x: x[1])
        x, y = positions[1][0], positions[1][1]
        prob_x = probabilities[0][x] * 100
        prob_y = probabilities[0][y] * 100
        formatted_prob_x = f"{prob_x:.2f}%"
        formatted_prob_y = f"{prob_y:.2f}%"
        print('############################################################################################\n')
        print(f"The model believes this sound is {classes[x]} with {formatted_prob_x} confidence,\n")
        print(f"and {classes[y]} with {formatted_prob_y} confidence\n")
        print('############################################################################################')


    else:
        positions = sorted(positions, key=lambda x: x[1])
        x, y, z = positions[1][0], positions[1][1], positions[1][2]
        prob_x = probabilities[0][x] * 100
        prob_y = probabilities[0][y] * 100
        prob_z = probabilities[0][z] * 100
        formatted_prob_x = f"{prob_x:.2f}%"
        formatted_prob_y = f"{prob_y:.2f}%"
        formatted_prob_z = f"{prob_z:.2f}%"
        print('############################################################################################\n')
        print(f"The model believes this sound is {classes[x]} with {formatted_prob_x} confidence,\n")
        print(f"{classes[y]} with {formatted_prob_y} confidence,\n")
        print(f"and {classes[z]} with {formatted_prob_z} confidence\n")
        print('############################################################################################')


def One_Sample_Prediction(filename, model_path, model_type):
    #Load the Model
    model = mt.model_load(model_path)
    clear_output()
    s,sr = librosa.load(filename,sr = 42000)
    #Hear the sound
    ipd.display(ipd.Audio(s, rate=sr))
    #Get the model prediction
    get_prediction(filename , model , model_type)

"""
The next oone is for a whole train set prediction

"""

def Train_Set_Predictions(model_name,model_type):

    
    
    if model_type == "YamNet":

        dataset = "YamNet"
        features_dir = "UrbanSound\\Features\\YamNet_Features\\Test_Features"
        x, y = dl.load_arrays_and_labels(features_dir)
        x, y, le = dl.datapreprocessing(x, y, dataset)

    else:

        dataset = "urbansound8k"
        # this specific file has problem it needs to be deleted
        ut.delete_files_with_prefix("UrbanSound\\Test", "36429")

        classes = ["children_playing", "drilling", "street_music", "siren", "gun_shot", "car_horn", "air_conditioner", "engine_idling",  "dog_bark", "jackhammer"]
        with tempfile.TemporaryDirectory() as tempdir:
            temp_test_dir = os.path.join(tempdir, "Test")
            temp_features_dir = os.path.join(tempdir, "Test_Features")
            os.makedirs(temp_test_dir, exist_ok=True)
            os.makedirs(temp_features_dir, exist_ok=True)
            
            # Copy the initial folder to the temporary directory
            shutil.copytree("UrbanSound\\Test", temp_test_dir, dirs_exist_ok=True)
            
            # Move files to corresponding category folders within the temporary test directory
            ut.move_to_class_folder("UrbanSound\\CSVs\\Test_csv", temp_test_dir, dataset)
            clear_output()
            
            if model_type == "CNN":
                # Extract features and save them to the temporary features directory
                fe.CNN_Features_mel(temp_test_dir, classes, temp_features_dir)
            else:
                fe.NN_Features(temp_test_dir, temp_features_dir)
            
            # Load and preprocess the data from the temporary features directory
            x, y = dl.load_arrays_and_labels(temp_features_dir)
            x, y, le = dl.datapreprocessing(x, y, dataset)
            
            # Print the names of the folders in the temporary directory
            print("Folders in the temporary directory:")
            for folder_name in os.listdir(tempdir):
                print(folder_name)
    
    # call the model
    model = mt.model_load(model_name)
    clear_output()
    
    #Make predictions
    mt.plot_confusion_matrix(model,x,y,le)

    
    