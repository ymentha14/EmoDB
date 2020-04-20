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
    │   ├── `visualisation.py`  # Methods to visualize the datas
    │   ├── `preprocess.py`     # Preprocessing methods (e.g. data augmentation etc)
    ├── lib                     # Modified libraries the project depends on
    │   ├── pytorch-cnn-viz     # Allows to generate class maximum output images
    ├── models                  # directory where we save the fitted models(.pkl,.json)
    │   ├── weights             # directory where we save the weights for  CNN_classif objects
    ├── data
    │   ├── mfcc                # mfcc converted files from wav directory
    │   ├── wav                 # wav files from the EmoDB dataset
    ├── docs                    # documentations
    ├── `Report_SER.ipynb`      # Complete analysis and results of the project
    ├── static                  # CSS files for the GUI
    ├── `build.sh`              # Bash executable to build the docker image
    ├── templates               # HTML files for the GUI
    ├── `requirements.txt`      # required libs for docker
    └── `README.md`

## Flask API commands
A GUI shows up if you open the http://127.0.0.1:5000/: enter the filename in the main field to obtain the prediction of the most recently compute d model.
**TrainModel**
----
**URL**: `/trainmodel`

**Method:**: `GET` 
  
**URL Params**: `n_epoch=[integer]`: number of epochs to run the model for

**Success Response Code:**   200 

**Error Response Code:** 201

**Return**: `None`

**Sample Call: **`curl http://127.0.0.1:5000/trainmodel?nb_epoch=1`

**Predict**
----
**URL**: `/predict`

**Method:**: `POST` 
  
**URL Params**: `filename=[string]`: name of the file to receive a prediction for.

**Success Response Code:**   200

**Error Response Code:** 201 

**Return**: {
  "predicted": &lt;derp predicted_emotion&gt;, 
  "true_label": "&lt;true_emotion&gt;"
}


**Sample Call: ** `curl -X POST http://127.0.0.1:5000/predict?filename=03a07La`


**PredictJSON**
----

**URL**: `/predict`

**Method:**: `POST` 
  
**URL Params**: `None`

**Data Params**: json file associating string values corresponding to the filenames to some keys. Example: <br>
{
	"key1":"03a04Wc",
	"key2":"12a02Ec",
  ...
}


**Success Response Code:**   200 

**Error Response Code:** 201 

**Return**: 

{
  "key1": {
    "predicted": "&lt;predicted_emotion1&gt;", 
    "true_label": "&lt;true_emotion1&gt;"
  }, 


  "key2": {
    "predicted": &lt;predicted_emotion2&gt;, 
    "true_label": "&lt;true_emotion2&gt;"
  }, 

  ...
}

**Sample Call:** `curl -X POST -d @data.json http://127.0.0.1:5000/predictJSON  --header "Content-Type:application/json"`
