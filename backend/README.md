# backend
Codebase for the OBS plugin backend for the quest tracker.

## Build - Windows
1. Make sure you have the following installed and added to path:
    ```
    cmake
    7z
    powershell 7
    git
    Visual Studio 17 2022
    ```
1. Open PowerShell 7, and run the installation script (only need to run once)
    ```
    scripts/Install-Repository.ps1
    ```
1. Build with the build script
    ```
    scripts/Build.ps1
    ```
1. The output will be in the `release` folder

## Build - Linux
Currently not available. If you figure out how to build it on linux, please let me know.

## Build - Mac
Currently not available. If you figure out how to build it on mac, please let me know.