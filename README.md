# Real-time Object Detection and Tracking

This is a computer vision application built for UWaterloo STAT 441 W18 final project.

## System Requirements
* python 3.5 and up

## Setup
A note on virtual environments: There are many tools for creating virtual environments, such as Anaconda, pipenv, and virtualenv.
You may find pipenv and virtualenv to be more light-weight solutions comparing with Anaconda. 
Creating a virtual environment to run the project is good for keeping dependencies and number of libraries used at a minimal, 
but is not necessary.

If you're using terminal:
* pip install opencv-python
* python
* import cv2 as cv

Alternatively, you can use homebrew:
* brew install opencv-python
* python
* import cv2 as cv

If you're using PyCharm:
* In Project Interpreter, select Python 3.5 from your local bin as the project interpreter. 
Then click on "+" in the package manager to add opencv-python to your site-packages directory.
* You should be able to import cv2 without any problems now.
