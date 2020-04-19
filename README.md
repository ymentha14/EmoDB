# EmoDB Speech Classification

The present repository aims at presenting a method allowing to perform emotion classification on the EmoDB dataset. This dataset contains samples of speech presenting one of 7 emotions: happiness, fear, anger, sadness, disgust, neutral or bordom.

## Getting Started


### Prerequisites

No particular prerequisites except docker.


### Installing

In order to build the docker image and download the data, run `build.sh` from the project root directory.

Once this is done, run `run.sh` to start a docker container. <br>

The jupyter lab instance and the flask API are respectively exposed on ports `8887` and `5000`.

In order to access the jupyter lab instance, you will need to copy-paste the token displayed on the output of the shell where you ran `run.sh` for security reason.

In order to access the GUI of the flask API, just open a tab on the `5000` port.

## Project Structure
 .
    ├── src                     # Source files
    │   ├── `misc_funcs.py`     # Miscellanous functions for data preparation/cleaning
    │   ├── `model.py`          # Classes and methods to perform the proper modelling
    ├── lib                     # Modified libraries the project depends on
    │   ├── pytorch-cnn-viz     # Allows to perform filter visualisation
    ├── models                  # directory where we save the fitted models
    ├── data        
    │   ├── mfcc                # mfcc converted files from wav directory
    │   ├── wav                 # wav files from the EmoDB dataset
    ├── `Report_SER.ipynb`      # Complete analysis and results of the project
    ├── static                  # CSS files for the GUI
    ├── `build.sh`              # Bash executable to build the docker image
    ├── templates               # HTML files for the GUI
    ├── `requirements.txt`      # required libs for docker
    └── `README.md`
