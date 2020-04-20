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

## Flask API commands
**TrainModel**
----
  <_Additional information about your API call. Try to use verbs that match both request type (fetching vs modifying) and plurality (one vs multiple)._>

* **URL**

  `/trainmodel`

* **Method:**
  
  `GET` 
  
*  **URL Params**

   `n_epochs=[integer]`: number of epochs to run the model for

* **Data Params**

  <_If making a post request, what should the body payload look like? URL Params rules apply here too._>

* **Success Response:**
  
  * **Code:** 200 <br />
* **Error Response:**
  * **Code:** 201  <br />

* **Sample Call:**

  `curl -GET localhost:5000`
* **Notes:**

  <_This is where all uncertainties, commentary, discussion etc. can go. I recommend timestamping and identifying oneself when leaving comments here._> 
**Predict**
----
  <_Additional information about your API call. Try to use verbs that match both request type (fetching vs modifying) and plurality (one vs multiple)._>

* **URL**

  `/trainmodel`

* **Method:**
  
  `GET` 
  
*  **URL Params**

   `n_epochs=[integer]`: number of epochs to run the model for

* **Data Params**

  <_If making a post request, what should the body payload look like? URL Params rules apply here too._>

* **Success Response:**
  
  <_What should the status code be on success and is there any returned data? This is useful when people need to to know what their callbacks should expect!_>

  * **Code:** 200 <br />
    **Content:** `{ id : 12 }`
 
* **Error Response:**

  <_Most endpoints will have many ways they can fail. From unauthorized access, to wrongful parameters etc. All of those should be liste d here. It might seem repetitive, but it helps prevent assumptions from being made where they should be._>

  * **Code:** 401 UNAUTHORIZED <br />
    **Content:** `{ error : "Log in" }`

  OR

  * **Code:** 422 UNPROCESSABLE ENTRY <br />
    **Content:** `{ error : "Email Invalid" }`

* **Sample Call:**

  <_Just a sample call to your endpoint in a runnable format ($.ajax call or a curl request) - this makes life easier and more predictable._> 

* **Notes:**

  <_This is where all uncertainties, commentary, discussion etc. can go. I recommend timestamping and identifying oneself when leaving comments here._> 
