"""
/src/misc-funcs.py: 
miscellanous functions for loading, treating and transforming the data
"""
import numpy as np
import os
from IPython.core.debugger import set_trace
from scipy.io import wavfile
from pathlib import Path
import pickle
from speechpy.feature import mfcc
import pandas as pd
import librosa

#############   directories paths   #############
MAIN_DIR = "./"
DATA_DIR = os.path.join(MAIN_DIR,"data")
WAV_DIR = os.path.join(DATA_DIR,"wav")
MFCC_DIR  = os.path.join(DATA_DIR,"mfcc")
MODEL_DIR = os.path.join(MAIN_DIR,"models")
WEIGHT_DIR = os.path.join(MODEL_DIR,"weights")


#############   label dictionaries   #############
DE2EN = {'W':'A', #Wut-Anger
         'L':'B', #Langeweile-Bordom
         'E':'D', #Ekel-Disgust
         'A':'F', #Angst-Fear
         'F':'H', #Freude-Happiness
         'T':'S', #Traueer-Sadness
         'N':'N'} #Neutral
EN2DE = {value:key for key,value in DE2EN.items()}
EN2NUM = {item[1]:num for item,num in zip(DE2EN.items(),range(len(DE2EN)))}
NUM2EN = {value:key for key,value in EN2NUM.items()}
FULL_EM = {'A':'Anger',
          'B': 'Bordom',
          'D':'Disgust',
          'F':'Fear',
          'H':'Happiness',
          'S':'Sadness',
          'N':'Neutral'}
DE2NUM = {item[0]:num for item,num in zip(DE2EN.items(),range(len(DE2EN)))}

#############   Speakers Metadatas   #############
SPEAKER_DATA = [[3  , 'male',  31],
                [8  , 'female',34 ],
                [9  , 'female',21 ],
                [10 , 'male',  32 ],
                [11 , 'male',  26 ],
                [12 , 'male',  30] ,
                [13 , 'female',32], 
                [14 , 'female',35] ,
                [15 , 'male',  25] ,
                [16 , 'female',31]]

def parse_filename(filename):
    """
    parses the attributes of a given sample based on its filename
    Args:
        filename(str):filename of the wav file (ex:"03a01Fa")
    Returns:
        speaker_id(int):id of the speaker
        text_id(str): id of the read text
        emotion_en(str): char describing the english emotion
    """
    speaker_id = int(filename[:2])
    text_id = filename[2:5]
    emotion_de = filename[5]
    emotion_en = DE2EN[emotion_de]
    return speaker_id,text_id,emotion_en

def load_pd_data(wav_dir=WAV_DIR):
    """
    Load the wav data with its metadats into a pandas DataFrame
    Args:
        wav_dir(str):path to the wav directory
    Returns:
        res(pd.DataFrame): pandas dataframe with metadatas cols=[speakerid,textid,emotion,data,sex,age]
    """
    for root, dirs, files in os.walk(wav_dir, topdown=False):
        paths = [os.path.join(root,file) for file in files]
        data = []
        for file in files:
            audio_data = wavfile.read(os.path.join(root,file))[1]
            speaker_id,text_id,emotion_en = parse_filename(file)
            row = [speaker_id,text_id,emotion_en,audio_data]
            data.append(row)
    res = pd.DataFrame(data,columns=["speaker_id","text_id","emotion","data"])
    speaker_data = pd.DataFrame(SPEAKER_DATA,columns = ['speaker_id','sex','age'])
    speaker_data.set_index('speaker_id',inplace=True)
    return res.join(speaker_data,on="speaker_id")

def zeropadd(data,mode='max'):
    """
    zero padds the audio files
    Args:
        mode(str):'max' or 'mean' if set to max, zero padds all the files to the max length of these files. Otherwise zero padds/cuts to the mean size.
        data(np.array): audio data
    Returns:
        data_padded(int): same data as in the arg, but zeropadded
    """
    if mode == 'max':
        new_len = max([x.shape[0] for x in data])
    else:
        new_len = int(np.round(np.mean([x.shape[0] for x in data])*1.5))
    def padd(x):
        diff = abs(new_len - x.shape[0])
        shift = diff %2
        diff //=2
        if x.shape[0] < new_len:
            return np.pad(x,(diff,diff+shift),'constant')
        else:
            return x[diff:-(diff+shift)]
    data_padded = np.zeros((len(data),new_len))
    for i,x in enumerate(data):
        data_padded[i] = padd(x)
    return data_padded

def load_wav_data(wav_dir=WAV_DIR):
    """
    load the wav data
    Args:
        wav_dir(str):path to the wav directory
    Returns:
        file_names(np.array(str)): names of the loaded files
        sfs(np.array(int)): frequencies of the audio data
        data(np.array): audio files
        targets(np.array): targets of the audio files
    """
    data,sfs,targets,file_names = [],[],[],[]
    for root, dirs, files in os.walk(wav_dir, topdown=False):
        for file in files:
            sf,audio_data = wavfile.read(os.path.join(root,file))
            data.append(audio_data)
            sfs.append(sf)
            target = DE2NUM[file[5].capitalize()]
            targets.append(target)
            file_names.append(file.split('.')[0])
    data = zeropadd(data,mode='mean')
    file_names = np.array(file_names)
    sfs = np.array(sfs)
    targets = np.array(targets)
    order = np.argsort(file_names)
    return file_names[order],sfs[order],data[order],targets[order]

def get_mfcc(data,sfs):
    """
    load the wav data
    Args:
        data(np.array): audio files
        sfs(np.array(int)): frequencies of the audio data
    Returns:
        (np.array): mel-frequency cepstrum of the audio data
    """
    if isinstance(sfs,(int,np.int64)):
        sfs = [sfs for i in range(len(data))]
    ret = np.array([mfcc(x,sf,num_cepstral=39) for x,sf in zip(data,sfs)])
    return np.expand_dims(ret,axis=1)
    

def save_mfcc_data(file_names,data_f,targets):
    """
    create mfcc directory and save the mfcc data for quicker load
    Args:
        file_names(np.array(str)): names of the loaded files
        data_f(np.array): mfcc files
        targets(np.array): targets of the audio files
    """
    Path("data/mfcc").mkdir(parents=False, exist_ok=True)
    for file,smple,target in zip(file_names,data_f,targets):
        file_name = file +".pkl"
        with open(os.path.join("data/mfcc",file_name),'wb') as f:
            save_data = (smple,target)
            pickle.dump(save_data,f)
    print("Saving done!")

def load_mfcc_data(mfcc_dir):
    """
    load previously saved mfcc data 
    Args:
        mfcc_dir(str): path of the directory of the mfcc data
    """
    data = []
    targets = []
    filenames = []
    for root, dirs, files in os.walk(mfcc_dir, topdown=False):
        for file in files:
            with open(os.path.join(root,file),'rb') as f:
                temp = pickle.load(f)
            data.append(temp[0])
            targets.append(temp[1])
            filenames.append(file.split('.')[0])
    data = np.array(data)
    targets = np.array(targets)
    filenames = np.array(filenames)
    order = np.argsort(filenames)
    return filenames[order],data[order],targets[order]

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        print("change")
        DATA_DIR = "./data"
        WAV_DIR = os.path.join(DATA_DIR,"wav")
        assert(os.path.isdir(DATA_DIR))
    file_names,sfs,data,targets = load_wav_data()
    data_f = get_mfcc(data,sfs)
    save_mfcc_data(file_names,data_f,targets)
    file_names,data_f,targets = load_mfcc_data(MFCC_DIR)