Jumper Studio
=============

Requirements
------------
- Windows 10/11 (64-bit)
- Python 3.11-3.13 (3.11 recommended) from python.org, with
  "Add python.exe to PATH". Nothing else needs to be preinstalled:
  MuJoCo, PySide6, Sports2D, OpenCV, ... are installed into the project's
  .venv by setup_windows.bat
- An NVIDIA GPU is required for PPO training (the app itself runs without one)
- Internet for the first setup and the first video analysis; offline afterwards

First run
---------
Double-click:

    start_windows.bat

It creates the app environment, downloads the Jumper training workspace
(D:\jumper_ws by default), installs the GPU stack and creates the desktop
shortcut. Follow the console; the first setup takes a while.

After setup, launch from the desktop shortcut "Jumper Studio" or run:

    start_windows.bat

Useful
------
- setup_windows.bat -Action diagnose   check the environment
- uninstall_windows.bat                remove selected parts
- AGENTS.md                            maintainer/agent specification
- README.md                            full feature and training documentation
- SETUP_WINDOWS.md                     manual setup and troubleshooting
