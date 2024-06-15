import pandas as pd
import json
import os
import shutil
from pydub import AudioSegment

def create_labels(input_csv, ontology_json, output_csv):

    
    """
    This function processes the CSV file to replace encoded labels with real names based on the ontology.

    Parameters:
    - input_csv: Path to the input CSV file.
    - ontology_json: Path to the ontology JSON file.
    - output_csv: Path to save the output CSV file with real label names.
    """
    
    # Read the CSV file
    df = pd.read_csv(input_csv, skiprows=2, delimiter=',', quotechar='"', on_bad_lines='skip')

    # List of column names from which to extract non-null values
    class_columns = [
        'positive_labels', 'class 2', 'class 3', 'class 4', 'class 5',
        'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 
        'class 12', 'class 13'
    ]

    # Iterate over each row to collect non-null values
    for index, row in df.iterrows():
        non_null_values = []
        for class_column in class_columns:
            if pd.notnull(row[class_column]):
                non_null_values.append(row[class_column])
        df.at[index, 'labels'] = ','.join(non_null_values)
    
    # Drop the original class columns
    df = df.drop(columns=class_columns)

    # Load ontology data to map label IDs to their names
    with open(ontology_json, 'r') as f:
        ontology_data = json.load(f)

    label_map = {item['id']: item['name'] for item in ontology_data}

    # Function to map label IDs to their names
    def map_labels(label_ids):
        labels = label_ids.split(',')
        names = [label_map[label_id.strip()] for label_id in labels]
        return ','.join(names)

    # Apply the mapping function
    df['labels'] = df['labels'].str.replace('"', '')
    df['label_names'] = df['labels'].apply(map_labels)

    # Save the processed CSV to the output path
    return df



"""

This script creates new csv only for the files we succesfully download and take their labels a lot of them where not downloaded due to copyroghtsd or errors
We use label_to_use to match the csv of the files we dowload with the labels

"""

def read_labels_csv(csv_file):
    df = pd.read_csv(csv_file)
    ytid_label_dict = {row['# YTID']: row['label_names'] for index, row in df.iterrows()}
    return ytid_label_dict

# Function to process files in the folder and match with labels
def labels_to_use(folder_path, labels_csv, output_csv):
    ytid_label_dict = read_labels_csv(labels_csv)
    matched_files = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            ytid = os.path.splitext(filename)[0]  # Extract YTID from filename
            if ytid in ytid_label_dict:
                labels = ytid_label_dict[ytid]
                matched_files.append({'Filename': filename, 'Labels': labels})
            else:
                print(f"No match found for file: {filename}")

    # Convert matched files to DataFrame and write to output CSV
    df_matched_files = pd.DataFrame(matched_files)
    df_matched_files.to_csv(output_csv, index=False)





def cut_audio(csv_path, audio_folder, output_folder):
    """
    This function cuts audio segments based on start and end times provided in a CSV file and exports them to the specified folder.
    
    Parameters:
    - csv_path: Path to the input CSV file containing segment information.
    - audio_folder: Path to the folder containing flagged audio files.
    - output_folder: Path to the folder where the exported segments will be saved.
    
    
    """
    
    # Read the CSV file
    segments = pd.read_csv(csv_path)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    """
    We have very specific columns in the csv file that we need to use to cut the audio files if your code doesn not find it change it


    """
    # Iterate through each row in the CSV file
    for index, row in segments.iterrows():
        filename = row['# YTID']
        start_time = int(row[' start_seconds']) * 1000  # Convert to milliseconds
        end_time = int(row[' end_seconds']) * 1000  # Convert to milliseconds
        
        # Construct full path to the audio file
        audio_file_path = os.path.join(audio_folder, f"{filename}.mp3")
        
        # Debugging statement to check file path and existence
        print(f"Processing file: {audio_file_path}")
        
        # Check if the file exists
        if not os.path.exists(audio_file_path):
            print(f"File {audio_file_path} not found. Skipping...")
            continue
        
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_file_path)
            
            # Cut the audio segment
            segment = audio[start_time:end_time]
            
            # Export the segment to a new file
            output_filename = os.path.join(output_folder, f"{filename}.mp3")
            segment.export(output_filename, format="mp3")
            
            print(f"Exported {output_filename}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print("All segments have been successfully exported.")



def convert_audio(input_folder, output_folder):
    """

    This function converts audio files in the input folder to a format that can be read by librosa and saves them in the output folder.
    
    Parameters:
    - input_folder: Path to the input folder containing audio files.
    - output_folder: Path to the output folder where processed files will be saved.
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            # Create the output filename
            output_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.mp3")
            
            # Check if the output file already exists, if yes, skip
            if os.path.exists(output_filename):
                print(f"File {output_filename} already exists. Skipping...")
                continue
            
            # Load the audio file
            audio = AudioSegment.from_file(os.path.join(input_folder, filename))
            
            # Export the entire audio to the new file
            audio.export(output_filename, format="mp3")
            
            print(f"Exported {output_filename}")
            
            # Debugging statements
            print(f"Input filename: {filename}")
            print(f"Output filename: {output_filename}")
            print()
    
    print("All files processed and exported.")
