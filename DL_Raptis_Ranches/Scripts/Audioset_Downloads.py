import os
import csv
import subprocess



# Check if yt-dlp is installed correctly (you can add more robust checks if needed)


try:
    import yt_dlp as youtube_dl
except ImportError:
    raise EnvironmentError(
        
        
        """
        These commands are MANDATORY TO BE RUN IN COLAB or in the cloud environment where you are downloading the dataset, or else you get an error:
        __________________________________________________________________________________________________
        
        !git clone https://github.com/lukefahr/audioset.git
        
        __________________________________________________________________________________________________
        
        !python3 -m pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
        __________________________________________________________________________________________________
        """
        
    )






def download_and_cut_audioset(train_csv_path, output_dir, flag_file=None):
    """
    This function cuts and downloads the files based on the CSV. 
    A lot of files may not be cut for some reason and can be cut later.
    
    Parameters:
    - train_csv_path: Path to the CSV file containing segment information.
    - output_dir: Directory where the audio files will be saved.
    - flag_file: Optional. File to start downloading from a specific segment ID.

    
    """
    
    def download_audio(segment_id, start_time, end_time, output_file):
        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping download.")
            return

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_file,
            'postprocessor_args': ['-ss', str(start_time), '-to', str(end_time)],
            'extractaudio': True,
            'audioformat': 'mp3',  # Change audio format to MP3
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download(['https://www.youtube.com/watch?v=' + segment_id])
            except youtube_dl.utils.DownloadError as e:
                print(f"Skipping segment {segment_id} due to error: {str(e)}")
                pass  # Skip this video and continue to the next one

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read train CSV file
    start_downloading = flag_file is None  # If no flag_file is provided, start downloading immediately

    with open(train_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            segment_id = row[0]  # Assuming the segment ID is in the first column
            start_time = row[1]   # Assuming start time is in the second column
            end_time = row[2]     # Assuming end time is in the third column

            # Check if the segment ID matches the specified one to start downloading
            if flag_file and segment_id == flag_file:
                start_downloading = True

            # If start_downloading is True, start downloading
            if start_downloading:
                # Download audio segment and save directly to the Audioset folder
                # Modify the output file extension to .mp3
                output_file = os.path.join(output_dir, f"{segment_id}.mp3")
                download_audio(segment_id, start_time, end_time, output_file)
                print(f"Downloaded segment {segment_id} to {output_file}")


