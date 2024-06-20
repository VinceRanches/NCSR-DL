import pandas as pd
import json
import os
from pydub import AudioSegment

"""
The next two functions creaste our labels

"""

def create_labels(input_csv, ontology_json):
    """
    This function processes the CSV file to replace encoded labels with real names based on the ontology.

    Parameters:
    - input_csv: Path to the input CSV file.
    - ontology_json: Path to the ontology JSON file.

    Returns:
    - df: DataFrame with the processed data
    """
    
   
    df = pd.read_csv(input_csv, skiprows=2, delimiter=',', quotechar='"', on_bad_lines='skip')

    # extract non-null values
    class_columns = [
        'positive_labels', 'class 2', 'class 3', 'class 4', 'class 5',
        'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 
        'class 12', 'class 13'
    ]

    # group the labels in a column
    df['labels'] = df[class_columns].apply(lambda row: ','.join(row.dropna().astype(str)), axis=1)
    df = df.drop(columns=class_columns)

    # map ontology to their labels
    with open(ontology_json, 'r') as f:
        ontology_data = json.load(f)

    label_map = {item['id']: item['name'] for item in ontology_data}

    # get the names
    def map_labels(label_ids):
        labels = label_ids.split(',')
        names = [label_map.get(label_id.strip(), label_id) for label_id in labels]
        return ','.join(names)

    # combine and keep only what we have
    df['labels'] = df['labels'].str.replace('"', '')
    df['label_names'] = df['labels'].apply(map_labels)

    return df

def get_labels(input_csv, ontology_json, output_csv, sound_path):
    """
    This function processes the input CSV and ontology JSON, matches the labels with the files in the specified folder,
    and saves the output to a new CSV file.

    Parameters:
    - input_csv: Path to the input CSV file.
    - ontology_json: Path to the ontology JSON file.
    - output_csv: Path to save the output CSV file with matched labels.
    - sound_path: Path to the folder containing sound files.
    """

    
    df = create_labels(input_csv, ontology_json)

    # get the matces
    ytid_label_dict = {row['# YTID']: row['label_names'] for _, row in df.iterrows()}

    matched_files = []

    # Iterate through files in the folder
    for filename in os.listdir(sound_path):
        if os.path.isfile(os.path.join(sound_path, filename)):
            ytid = os.path.splitext(filename)[0]  # Extract YTID from filename
            labels = ytid_label_dict.get(ytid)
            if labels:
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
    
    segments = pd.read_csv(csv_path)
    os.makedirs(output_folder, exist_ok=True)

    """
    We have very specific columns in the csv file that we need to use to cut the audio files if your code doesn not find it change it


    """
    
    for index, row in segments.iterrows():
        filename = row['# YTID']
        start_time = int(row[' start_seconds']) * 1000  
        end_time = int(row[' end_seconds']) * 1000  
        
        # get full path
        audio_file_path = os.path.join(audio_folder, f"{filename}.mp3")
        
        # check if the file exists
        print(f"Processing file: {audio_file_path}")

        if not os.path.exists(audio_file_path):
            print(f"File {audio_file_path} not found. Skipping...")
            continue
        
        try:
            # error handling
            audio = AudioSegment.from_file(audio_file_path)
            
            segment = audio[start_time:end_time]
            
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
    
    os.makedirs(output_folder, exist_ok=True)
    # see all files
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
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
