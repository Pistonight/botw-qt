# cnn
A Convolutional Neural Network (CNN) trained for detecting quest banner images for All Quests Speedruns in The Legend of Zelda: Breath of the Wild.

**NOTE: The model is not trained with DLC quests. Be careful when checking data**

## Table of Contents
1. [Setup](#setup)
    1. [CUDA and cuDNN](#cuda-and-cudnn)
    2. [TensorFlow](#tensorflow)
2. [Scraping](#scraping)
3. [Training](#training)
4. [Evaluating](#evaluating)

## Use Guide
Will be available soon

## Setup
To setup the repo for scraping or training, please follow these steps.

### CUDA and cuDNN
If you have an NVIDIA GPU compatible with CUDA 11, you can setup CUDA for faster training. Not required if you are scraping or plan to train with CPU.

The steps below covers how to setup CUDA with TensorFlow in WSL 2. If you are running another operating system, please refer to the official documentation.

Note that TensorFlow 2.11 is only compatible with CUDA 11, not 12.

1. Prerequisite: Have WSL 2 installed and running, and have the latest Nvidia Game Ready drivers installed in Windows
1. Download the CUDA Toolkit 11.8.0 from [Nvidia](https://developer.nvidia.com/cuda-toolkit-archive). You need to sign up for the Nvidia Developer Program to download the toolkit. **Select `Linux` - `x86_64` - `WSL-Ubuntu`** for the architecture.
1. Once installed, you should have CUDA at `/usr/local/cuda-11.8` and a symlink at `/usr/local/cuda`
1. Download cuDNN from [Nvidia](https://developer.nvidia.com/rdp/cudnn-download) for CUDA 11.x. **Select Local Installer for Linux x86_64 (Tar)**
1. Extract the downloaded file
    ```
    tar -xf cudnn-linux-x64-8.8.0.121_cuda11-archive.tar.xz
    ```
1. Copy the include and lib to the cuda directory (Replace `<path>` with the path to the extracted files)
    ```
    sudo cp -P <path>/include/* /usr/local/cuda/include
    sudo cp -P <path>/lib64/* /usr/local/cuda/lib64
    ```
1. Run `sudo ldconfig`
1. Run `ldconfig -p | grep libcuda`. Make sure you have `libcuda.so.1` and `libcudart.so.11` (or `libcudart.so.11.0`) in the output
1. Run `ldconfig -p | grep libcudnn`. Make sure you have `libcudnn.so.8` in the output

### TensorFlow
If you plan to use GPU to train, follow the steps above to setup CUDA and cuDNN before setting up TensorFlow.

1. Install the requirements (setup a venv if wanted)
    ```
    pip install -r requirements.txt
    ```
1. To check if you are using CPU or GPU with TensorFlow, run the following
    ```
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
    ```
    You should see your CPU, and if you have CUDA setup, you should see your GPU as well.

## Scraping
Instructions for scraping data from VODs

### Requirements
1. Setup TensorFlow (see above)
1. Download the VOD
1. Download the latest tflite model from the release page

### Steps
**Instructions are incomplete/outdated**
`data_scrape.py` and `data_none.py` are used for dataset generation from VODs.

1. Run `data_scrape.py` to generate images from the VOD. Replace `<dataset_name>` with `<runner><time>`, such as `piston820` or `bings1532`. Use `-j` to specify the number of processes to use.
    ```
    python data_scrape.py -j 16 --model path/to/model --video path/to/vod --raw raw/<dataset_name>
    ```
    If you need to crop the vod:
    ```
    python data_scrape.py -j 16 --model path/to/model --video path/to/vod --raw raw/<dataset_name> --rect "<x>,<y>,<w>,<h>"
    ```
    
1. Manually check the images labelled with a quest in `raw/<dataset_name>_unchecked` to make sure the images are correct. Move the images labelled with a quest to `raw/<dataset_name>`. See [Manual Scrubbing](#manual-scrubbing) for more details.

### Manual Scrubbing
**Instructions are incomplete/outdated**
All images must be manually checked before used for training or validation. After filtering the data, manually check if any images are mislabelled. Move any mislabeled data to the correct quest folder.

The rule of thumb when labeling data is, if you can tell the quest from the image by itself (i.e. not looking at frames around it), then label the image with the quest. Otherwise label it none.

For example, if you see the word "CHECK", but all other things are unreadable, you know it's "THE GUT CHECK CHALLENGE" since that's the only quest with the word "CHECK" in it. So you label the image with "THE GUT CHECK CHALLENGE".

DO NOT use the model to predict before you make the decision, since that will influence you.

Another example: If there are 2 consecutive frames, and you can tell the first one is "UNDER A RED MOON". The second one is unreadable, but you know it's also "UNDER A RED MOON". In this case, label the first one "UNDER A RED MOON", and the second one "none".

Be especially careful when checking the images labeled with a quest, since good banner images will influence you when looking at the bad ones.

## Training
**Instructions are incomplete/outdated**
Training the model requires at least 2 datasets (1 for training and 1 for validation). The datasets are generated with the [scraping](#scraping) process and imported using the `import.py` script.

1. Modify `train.py` with the parameters you want to use, and also modify the model if needed
2. Run `train.py` with the datasets. The last dataset is used for validation
    ```
    python train.py <dataset 1> <dataset 2> <dataset 3> ...
    ```
3. The model will be saved to `models/<seed>.tflite` with additional information in `json` format alongside the model

## Evaluating
TODO