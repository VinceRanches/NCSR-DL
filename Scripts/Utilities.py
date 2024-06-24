import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv

# Used for flagging files after download

def flag_old_files(folder_path, cutoff_date):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_mtime = os.path.getmtime(file_path)
            if file_mtime < cutoff_date:
                new_file_name = 'FLAG-' + file
                new_file_path = os.path.join(root, new_file_name)
                os.rename(file_path, new_file_path)
                print(f'Renamed: {file_path} to {new_file_path}')


"""
Move the files that are flagged to a folder with prefix FLAG also you need to put these manually so we can edit them
on the Big_File_cut.ipynb

"""


def move_files_with_prefix(source_folder, target_folder, prefix):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Move files with the specified prefix to the target folder
    for file in files:
        if file.startswith(prefix):
            source_path = os.path.join(source_folder, file)
            target_path = os.path.join(target_folder, file)
            shutil.move(source_path, target_path)
            print(f"Moved {file} to {target_folder}")

"""
We need this before extracting features for aur NN with PyAudioAnalysis cause some of them have problems
these are the files

children_playing_folder = 'Train/children_playing'
drilling_folder = 'Train/drilling'
jackhammer_folder = 'Train/jackhammer'


children_playing_prefix = '36429'
drilling_prefix = '19007'
jackhammer_prefix = '88466'


delete_files_with_prefix(children_playing_folder, children_playing_prefix)
delete_files_with_prefix(drilling_folder, drilling_prefix)
delete_files_with_prefix(jackhammer_folder, jackhammer_prefix)


"""
def delete_files_with_prefix(folder, prefix):
    for filename in os.listdir(folder):
        if filename.startswith(prefix):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


"""
This script and the next one are used for debugging

"""


def next_file(folder_path, target_file):
    found_target = False

    # Iterate over all files and folders in the specified folder
    for item in os.listdir(folder_path):
        if item == target_file:
            found_target = True
        elif found_target:
            print("Next file after {}: {}".format(target_file, item))
            break
    
    if not found_target:
        print("File '{}' not found in folder '{}'.".format(target_file, folder_path))

# # Example usage
# folder_path = "/content/drive/MyDrive/Colab Notebooks/data/DL_Project/Audioset/"
# target_file = "kJT6o3gRQgQ.mp3"
# find_next_file(folder_path, target_file)



def exist_and_delete(file_path):# Check if the file exists before attempting to delete it
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    else:
        print(f"File '{file_path}' does not exist.")


"""
These two functions create the folders and categories for each file it can be applied to both audioset and urban sound
dataset is a list with the names of the classes for example 

#dataset = "audioset"
dataset = "urbansound8k"

if dataset == "audioset":

    classes = ["Animal Sounds","Human Sounds","Musical Instruments", "Environmental Sounds", "Vehicle Sounds", "Machine and Tool Sounds", "Impact Sounds", "Miscellaneous"]

else:
    
    classes = ["children_playing", "drilling", "street_music", "siren", "gun_shot", "car_horn", "air_conditioner", "engine_idling",  "dog_bark", "jackhammer"]

    use it when you want to make files 

"""

def move_to_class_folder(csv_file,folder_path, dataset):
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(csv_file)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
          
        # Construct the WAV file name from ID
        
        if dataset == 'urbansound8k':
            category = row['class']
            file_id = str(row['slice_file_name'])
        else:
            category = str(row['Category'])
            file_id = str(row['Filename'])
        
        # Check if the WAV file exists
        if os.path.exists(os.path.join(folder_path, file_id)):
            # Create folder if it doesn't exist
            category_folder = os.path.join(folder_path, category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            
            # Move the WAV file to the corresponding category folder
            shutil.move(os.path.join(folder_path, file_id), os.path.join(category_folder, file_id))
            print(f"Moved {file_id} to {category} folder.")
        else:
            print(f"File {file_id} not found.")

    print("Task completed.")

"""

-----------------This is only for urbansound to separate the files into train test folder---------------


"""


def move_train_test_folder(csv_file, destination_folder, audio_dir):
    """ 
    
    Takes input where the csv is located on train and where the test ,  
                                                      where you want to save them , 
                                                      where all audio files are located
    """

    root_dir = audio_dir
    #
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Read the CSV file and create a dictionary with slice_file_name as the key and folder as value
    wav_mapping = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            wav_mapping[row['slice_file_name']] = row['fold']

    # Iterate through each folder in the root directory
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Check if there is a WAV file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    wav_file = os.path.join(folder_path, file_name)
                    # Check if the WAV file matches a slice_file_name in the CSV
                    if file_name in wav_mapping:
                        # Move the WAV file to the destination folder
                        shutil.move(wav_file, destination_folder)
                        print(f"Moved {file_name} to {destination_folder}")




""" Read the csv and then this will make a train test split to use """

def split_8k_train_test(input_csv,  train_output_csv,  test_output_csv):
    # Read the original CSV file
    df = pd.read_csv(input_csv)
    np_array = df.to_numpy()
    
    # Splitting features (X) and labels (y)
    X = np_array[:, :7]  # Columns 0 to 6
    y = np_array[:, 7]   # Column 7
    
    # Splitting into training and testing sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=18)
    
    # Reconstructing DataFrames for training and testing sets
    feature_names = ['slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'classID']
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['class'] = y_train
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['class'] = y_test
    
    train_dir = os.path.dirname(train_output_csv)
    test_dir = os.path.dirname(test_output_csv)
    
    if train_dir and not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    if test_dir and not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Saving to CSV files
    df_train.to_csv(train_output_csv, index=False)
    df_test.to_csv(test_output_csv, index=False)
