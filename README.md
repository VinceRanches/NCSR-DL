# Deep Learning Project Sound classification 

### Prerequisites

To run this project, you need to have Anaconda installed. You can download and install Anaconda from [here](https://www.anaconda.com/products/distribution).

Download the need features from [here](https://drive.google.com/drive/folders/1vToSq-XES05nR0M9EYnIx2RIOulwXXpi?usp=sharing). (see Directory structure)

### Setting Up the Environment

1. Create a Conda environment with Python 3.10:
    ```bash
    conda create -n env python=3.10
    ```

2. Activate the Conda environment:
    ```bash
    conda activate env
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Notebooks

Carefully read through the Jupyter notebooks and run them in order. The sequence is important for ensuring that all necessary steps are completed:

1. `1.Urban_Sound_Pipeline`
2. `2.Audioset_Pipeline`
3. `3.Transfer_Learning_Pipeline`

`4.Demo` is intended for demonstration purposes.

### Directory Structure

If you download features and models from Google Drive, ensure that the directory structure is arranged as follows:
```
Project
├── Audioset
│   ├── AS_Eval (Only if you download the audio)
│   ├── AS_Train (Only if you download the audio)
│   ├── CSVs
│   ├── Train_Features
│   └── Eval_Features
├── Bonus
├── Models
├── Notebooks
├── Scripts
└── UrbanSound
    ├── CSVs
    ├── Features
    │   ├── CNN_Features
    │   ├── NN_Features
    │   └── YamNet_Features
    ├── Test
    ├── Train
    └── UrbanSound8K (Original Folder)
```

