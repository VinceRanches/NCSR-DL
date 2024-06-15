import pandas as pd
import numpy as np

from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction as dW
from deep_audio_features.utils import sound_processing as sp
from deep_audio_features.bin.config import WINDOW_LENGTH, HOP_LENGTH

import os
import librosa



"""
This function takes as input the directory where you have your files separated into classes and uses deep audio

features library to extract and save melspectorgram features to a selected directory also it handles files which can not be read 

hop length and window size are the default deep audio features also when a file is smaller than 128,100 it gets zero 

padded so every feature has the same shape

Wherever you save them the folders should be Train_Features and Eval_Features correspondingly
"""

def CNN_Features_mel(directory, classes, output_directory):
    failed_files = []

    for class_name in classes:
        class_directory = os.path.join(directory, class_name)

        for file_name in os.listdir(class_directory):
            try:
                # Load audio file
                audio_file = os.path.join(class_directory, file_name)
                x, fs = librosa.load(audio_file, sr=None)

                win_length = int(WINDOW_LENGTH * fs)
                hop_length = int(HOP_LENGTH * fs)

                mels = sp.get_melspectrogram(x, fs, hop_length)
                mels = np.transpose(mels)

                mels = mels[:, :100]
                if mels.shape[1] < 100:
                    mels = np.pad(mels, ((0, 0), (0, 100 - mels.shape[1])), 'constant')

                # Specify the directory and filename
                save_directory = os.path.join(output_directory, class_name)
                os.makedirs(save_directory, exist_ok=True)
                wav_filename = file_name
                filename_without_ext, _ = os.path.splitext(wav_filename)

                # Save the Mel spectrogram to a binary file in the specified directory
                np.save(os.path.join(save_directory, filename_without_ext + '.npy'), mels)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                failed_files.append(file_name)

    # Print the names and count of the failed files
    if failed_files:
        print("\nFailed files:")
        for file_name in failed_files:
            print(file_name)
        print(f"\nTotal failed files: {len(failed_files)}")
    else:
        print("\nNo files failed.")

""" 

Takes as input the folder path which contains the folder of data where evry folder == class and returns the features in binary files
example use ------>   Features_Neural_Network = extract_features_NN('path\\Train_Folder',"path\\Train_NN_Features")    <------ 

!!! Some specific files for urbansound need to be deleted!!!


"""

def NN_Features(folder_path, output_folder):


    classes = os.listdir(folder_path)  # Get list of class (folder) names
    for folder_name in classes:
        class_path = os.path.join(folder_path, folder_name)
        # Assuming dW, _ functions are defined elsewhere
        f, _, fn = dW(class_path, 1, 1, 0.1, 0.1)  # Store features and corresponding feature names
        
        # Create a DataFrame from the NumPy array
        temp_df = pd.DataFrame(f, columns=fn)
        
        # Add class and file name information to DataFrame
        for i in range(len(_)):
            part = _[i].split(os.sep)  # Split the path using the appropriate separator
            class_cat = part[-2]  # Extract class category from path
            wav_file = part[-1]  # Extract file name from path
            wav_file = wav_file[:-4]
        
            #temp_df.loc[i, 'class'] = class_cat  # Optionally, add class category to DataFrame
            temp_df.loc[i, 'Filename'] = wav_file
        
        # Ensure the output folder for the class exists
        os.makedirs(os.path.join(output_folder, folder_name), exist_ok=True)
        
        # Save each row as a NumPy binary file
        for index, row in temp_df.iterrows():
            filename = row['Filename']  # Assuming 'Filename' is the column containing file names
            np.save(os.path.join(output_folder, folder_name, f"{filename}" + ".npy"), f[index])
    


