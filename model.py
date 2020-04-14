import os
from scipy.io import wavfile
DATA_DIR = "./data"
WAV_DIR = os.path.join(DATA_DIR,"wav")

DE2EN = {'W':'A', #Wut-Anger
         'L':'B', #Langeweile-Bordom
         'E':'D', #Ekel-Disgust
         'A':'F', #Angst-Fear
         'F':'H', #Freude-Happiness
         'T':'S',
         'N':'N'} #Traueer-Sadness
EN2DE = {item:key for key,item in DE2EN.items()}
FULL_EM = {'A':'Anger',
           'B':'Bordom',
           'D':'Disgust',
           'F':'Fear',
           'H':'Happiness',
           'S':'Sadness',
           'N':'Neutral'}

def run_model():
    """
    train and save the best model on the dataset
    """
    print("WE TRAIN THAT'S HARD")
    return True

def dummy_model(filename):
    data = wavfile.read(os.path.join(WAV_DIR,"{}.wav".format(filename)))[1]
    emotion = DE2EN[filename[5]]
    return {'true_label':FULL_EM[emotion],
            'predicted': FULL_EM[emotion]}