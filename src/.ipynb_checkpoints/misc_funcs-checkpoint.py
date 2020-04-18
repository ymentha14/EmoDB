import numpy as np
import os
from IPython.core.debugger import set_trace
from scipy.io import wavfile
from pathlib import Path
import pickle
from speechpy.feature import mfcc

MAIN_DIR = "./"
DATA_DIR = os.path.join(MAIN_DIR,"data")
WAV_DIR = os.path.join(DATA_DIR,"wav")
MFCC_DIR  = os.path.join(DATA_DIR,"mfcc")
MODEL_DIR = os.path.join(MAIN_DIR,"models")

DE2EN = {'W':'A', #Wut-Anger
         'L':'B', #Langeweile-Bordom
         'E':'D', #Ekel-Disgust
         'A':'F', #Angst-Fear
         'F':'H', #Freude-Happiness
         'T':'S',
         'N':'N'} #Traueer-Sadness
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

def zeropadd(data,mode='max'):
    if mode == 'max':
        new_len = max([x.shape[0] for x in data])
    else:
        new_len = int(np.round(np.mean([x.shape[0] for x in data])))
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
    ret = np.array([mfcc(x,sf,num_cepstral=39) for x,sf in zip(data,sfs)])
    return np.expand_dims(ret,axis=1)

def save_mfcc_data(file_names,data_f,targets):
    Path("data/mfcc").mkdir(parents=False, exist_ok=True)
    for file,smple,target in zip(file_names,data_f,targets):
        file_name = file +".pkl"
        with open(os.path.join("data/mfcc",file_name),'wb') as f:
            save_data = (smple,target)
            pickle.dump(save_data,f)
    print("Saving done!")

def load_mfcc_data(mfcc_dir):
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
    print("hello",DATA_DIR)
    file_names,sfs,data,targets = load_wav_data()
    data_f = get_mfcc(data,sfs)
    save_mfcc_data(file_names,data_f,targets)
    file_names,data_f,targets = load_mfcc_data(MFCC_DIR)