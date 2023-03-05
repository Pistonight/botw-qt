# botw-qt
A system that trackers quests in BOTW speedruns.

The system includes 2 parts: an OBS plugin that reads the frames and a python program that detects if the frames contain quest banners.

Due to limitations, the program doesn't track quest completion after it's discovered. It only tracks the first time a quest is seen.

## Hardware
Check if your hardware supports the program before continuing:

### Operating System
The downloads only include release for Windows. If you use linux/mac and want to try the program, please let me know.

### Capture Card
Currently, the only supported video format is YUY2. Follow these steps to check if your capture card supports it:
1. Open OBS
2. Open the properties of your capture card source, and change "Resolution/FPS type" to "Custom". Then set the resolution, fps manually, see if "YUY2" is an option for "Video Format".
3. Revert the changes you made to the properties. Later, if the plugin isn't working with the default video format, you can change it to YUY2 manually here.

### OBS
Check if your OBS version matches the one stated in the release page.

## Installation
1. You need python 3.10.x installed in the system. You can download it from https://www.python.org/downloads/.
1. Download the 2 `zip` files from the [release](https://github.com/iTNTPiston/botw-qt/releases) page.
1. For the OBS plugin, extract the `zip` file and copy the 2 folders `data` and `obs-plugins` to your OBS directory, like you would when installing any obs plugin.
1. For the python program, extract the `zip` file. There's a `setup.bat` script that you can run to automatically install on windows. To manually install:
    ```
    pip install -r requirements.txt
    ```
    If you are on windows, also install `windows-curses`:
    ```
    pip install windows-curses
    ```

## OBS Configuration
Follow these steps to setup the OBS plugin to send frames to the python program:
1. Launch OBS
2. Open the filter settings for your source. The source must be a video source, such as a capture card or a VLC video source.
3. Under "Audio/Video Filters", Add "BotW Quest Tracker Backend"
4. Check "Enable Preview" if it's not already checked. You should see an overlay on the preview on top of where the quest banner would show up, like below
![Example image for quest banner preview](https://cdn.discordapp.com/attachments/951389021114871819/1081747993268588624/image.png)
    - If you don't see the overlay. Go to the capture card source properties, and change "Resolution/FPS type" to "Custom". Then set the resolution, fps manually, and change "Video Format" to "YUY2". If you don't see the "YUY2" option, please let me know on discord.
5. Once you have the preview working, you can hide the preview by unchecking "Enable Preview" in the filter settings. The filter will not use resources if the python program isn't connected, so you can leave it there even when you're not using it.

## Usage
If you installed the python program using the `setup.bat` script, you can launch it by running `start.bat`. Otherwise, you can launch it by running `python client.py` in a command prompt. You will see a window like below:
![Example image for tracker program](https://cdn.discordapp.com/attachments/951389021114871819/1081750943726579943/image.png)
- On the top, you can see a (low-quality) preview of the last image received from OBS
- Below that there are two lists for discovered and undiscovered quests. You can use the arrow keys to scroll through the lists, and use the Enter key to mark/unmark a quest manually.

If you don't see the image preview, don't worry. The program will not send an image if it's too bright or too dark to contain a quest banner. Try moving around in the game or try getting a quest.

Throughout the run, the program will automatically save the state to `state.json` when a quest is updated. You can delete `state.json` to reset the program to its initial state (all quests undiscovered).