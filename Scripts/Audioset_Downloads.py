import os
import csv
import subprocess



# Check if yt-dlp is installed correctly (you can add more robust checks if needed)


try:
    
    import yt_dlp as youtube_dl

except ImportError:

    raise EnvironmentError(
        """

        These commands are MANDATORY TO RUN in colab or whichever other cloud environment where you are downloading the dataset, or else you get an error:
        __________________________________________________________________________________________________________________________________________________
        
        Clonethis repo:

        !git clone https://github.com/lukefahr/audioset.git
        
        ___________________________________________________________________________________________________________________________________________________
        
        Install this yt-dlp package:

        !python3 -m pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
        ___________________________________________________________________________________________________________________________________________________

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
        
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping download.")
            return

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_file,
            'postprocessor_args': ['-ss', str(start_time), '-to', str(end_time)],
            'extractaudio': True,
            'audioformat': 'mp3',  
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download(['https://www.youtube.com/watch?v=' + segment_id])
            except youtube_dl.utils.DownloadError as e:
                print(f"Skipping segment {segment_id} due to error: {str(e)}")
                pass  

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    start_downloading = flag_file is None  

    with open(train_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            # This is where we have evything in CSV
            segment_id = row[0]  
            start_time = row[1]   
            end_time = row[2]     
            # Check if the segment ID matches the specified one to start downloading
            # In case omethign stops the proccess and you want to start from a specific file
            if flag_file and segment_id == flag_file:
                start_downloading = True

            # If start_downloading is True, start downloading
            if start_downloading:
                # Download audio segment and save directly to the Audioset folder
                
                output_file = os.path.join(output_dir, f"{segment_id}.mp3")
                download_audio(segment_id, start_time, end_time, output_file)
                print(f"Downloaded segment {segment_id} to {output_file}")


