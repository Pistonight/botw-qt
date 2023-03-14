# cnn
A Convolutional Neural Network (CNN) trained for detecting quest banner images for All Quests Speedruns in The Legend of Zelda: Breath of the Wild.

The goals of the model:
1. Predict with high confidence of the correct quest if a frame is a quest banner
2. Not predict the quest from a frame that is either not a quest banner, or is a different quest than what is predicted.

**NOTE: The model is not trained with DLC quests.**

## Table of Contents
1. [Setup](#setup)
2. [Scraping](#scraping)
3. [Training](#training)
4. [Evaluating](#evaluating)

## Setup
Python 3.10.9 is required for this project.

It's also recommended to use a python virtual environment (venv) for this project.

You can either use pyenv or the built-in venv module to setup a venv. pyenv is Recommened for Linux/Mac and the built-in venv is recommended for Windows.

**NOTE: You can install CUDA/cuDNN if you'd like. However, in my testing, the model is too small to benefit from GPU acceleration.**

- pyenv:
    ```
    pyenv virtualenv 3.10.9 botw-qt
    pyenv local botw-qt
    ```
    pyenv should automatically activate the venv based on the `.python-version` file. Or you can mnaually activate it with
    ```
    pyenv shell botw-qt
    ```
- built-in venv:
    ```
    python -m venv venv
    ```
    You need to manually activate the venv every time you open a new terminal:
    - Windows (Powershell):
        ```
        venv\Scripts\activate
        ```
    - Linux/Mac (Bash):
        ```
        source venv/bin/activate
        ```
Once you have python and optinally venv setup, install the requirements:
```
pip install -r requirements.txt
```


## Scraping
Instructions for scraping data from VODs

1. Download the VOD. Many formats are supported, such as `.webm`, `.mp4`, `.mkv`, etc. For best result, please either use 720p or 1080p. I recommend using [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading the VOD.
1. Have a `.tflite` model.
1. Run `run_scrape.py` to generate images from the VOD. Replace `<dataset_name>` with `<runner><time>`, such as `piston820` or `bings1532`. Use `-j` to specify the number of processes to use.
    ```
    python run_scrape.py -j 16 -m path/to/model --video path/to/vod -d data/<dataset_name>
    ```
    If you need to crop the vod:
    ```
    python run_scrape.py -j 16 -m path/to/model --video path/to/vod -d data/<dataset_name> --rect "<x>,<y>,<w>,<h>"
    ```
    This script does the following automatically:
    1. Grab sample frames from video, crop it down to the quest banner, and save it as `png` files. By default, it grabs every 3rd frame. The images are saved at `data/<dataset_name>.tmp`
    1. Remove images that are exactly the same
    1. Run the model on the images to predict the quest. The images are saved at `data/<dataset_name>/<quest>`. By default it uses a condifence threshold of 97.
1. Manually check the following and move the images accordingly
    1. `data/<dataset_name>/<quest>`: Make sure all quest banner images are good quality. Follow the [Checking the data](#checking-the-data) section for more details.
    2. `data/<dataset_name>.tmp`: Manually label these and move the images accordingly. Delete this folder once done.

### Checking the data
Images are split into 3 categories:
1. Good quality images (`<dataset>/<quest>`). These ones are clearly not a banner, or the quest can be clearly identified from the image. These can be used for both training and validation. Examples of good quality images:
    - Full banner
    - A banner with minor obstruction
    - Wider strokes and majority of ligher strokes are visible (these are almost partial, but can still be used for training)
2. Partial images (`<dataset>.par/<quest>`). These ones are banner images that are almost unable to be read, but the quest still identifiable. Some rules for determining if an image is partial:
    - Only wider strokes are visible
    - Large chunks of words are obstructed, but the quest can still be inferred (like `Di<unreadable>ris` is `Divine Beast Vah Naboris`)
    - Other reason that the quest can be inferred, but the image should not be used for training.
    
    Partial images are treated as not quest banners in training.
3. Bad quality images (`<dataset>/<none>`). These ones are banner images that are unreadable. These are used for both training and validation. For example:
    - The image looks like words, but a quest cannot be inferred
    - Critical parts of the quest are obstructed, and the quest cannot be inferred (like `Divine Beast Vah <unreadable>`)

If the runner has DLC installed, put the DLC banners at `<dataset>.dlc/none`. Currently, DLC quests are not supported by the model. These images are neither used for training nor validation.


## Training

### Data Configuration
The data configuration file `config/data.yaml` defines what data is used for training and validation.

In general, we are using the good quality images only for training, and the `*.par` and `*.dlc` images for validation. Additionally, we are keeping one set of good quality images for validation as well.

### Model Configuration
The model configuration file defines parameters for the model. See comments in the file for more details.

### Running
Use the `run_training.py` script to train a model with given data and model configurations. Example:
```
python run_training.py -c config/data.yaml -c config/mk7.yaml
```
The first `-c` is for the data configuration, and the second `-c` is for the model configuration.

The script will create a `models/` folder and save the model as `<seed>.tflite`. Additionally, a `<seed>.yaml` file will be saved with the data and model configurations used for training, along with validation results. This `yaml` file can be used in retraining and evaluation.

## Evaluating and Inspection

### Evaluating with a single dataset
This is useful for detecting if there's any mislabeled images in the dataset. Example:
```
python run_evaluate.py -d data/piston820 -m models/1234.tflite -c wrong.json -j 32
```
This will run the model on the `data/piston820` dataset, and save any inconsistencies to `wrong.json`. You can then run the script to inspect the wrongs and update the dataset according to [the rules above](#checking-the-data).
```
python run_inspect.py -c wrong.json
```

### Evaluating with multiple datasets
This can tell us how close the model is to our goal. We can use the yaml config files to evaluate multiple datasets. Example:
```
python run_evaluate.py -c wrong.json -c models/1234.yaml -m models/1234.tflite -j 32
```
This will run the model on the validation datasets defined in `models/1234.yaml`. To run on training datasets as well, add `-e use-training-data` to the command.

Focus on these metrics when evaluating the overall performance of the model:
|Metric|Description|Range|
|---|---|---|
|Min Safe Confidence Threshold|The lowest confidence level for the model to achieve goal #2 (i.e. not making a harmful wrong prediction). The lower this is, the better|0-100|
|Low/High/Median Quests|This can tell if the model is especially bad at some of the quests|0-1|
|Usefulness|This is the ratio of banner images detected correctly, over all banner images. The higher this is, the better|0-1|
|Accuracy|Like Usefulness, but also includes non-banner images. Generally this metric is not as useful as Usefulness|0-1|
