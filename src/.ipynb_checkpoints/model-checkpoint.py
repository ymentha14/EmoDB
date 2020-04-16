import os
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from IPython.display import clear_output
from IPython.core.debugger import set_trace

import numpy as np
from datetime import datetime
from misc_funcs import MFCC_DIR,MODEL_DIR,WAV_DIR,DE2EN,NUM2EN,FULL_EM,load_mfcc_data,get_mfcc


class CNN_classif(nn.Module):
    def __init__(self):
        super(CNN_classif,self).__init__()
        self.convblock1 = nn.Sequential(
                                nn.Conv2d(1,8,kernel_size=13),
                                nn.BatchNorm2d(8),
                                nn.ReLU())
        self.convblock2 = nn.Sequential(
                                nn.Conv2d(8,8,kernel_size=13),
                                nn.BatchNorm2d(8),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(2,1)))
        self.convblock3 = nn.Sequential(
                                nn.Conv2d(8,8,kernel_size=13),
                                nn.BatchNorm2d(8),
                                nn.ReLU())
        self.convblock4 = nn.Sequential(
                                nn.Conv2d(8,8,kernel_size=2),
                                nn.BatchNorm2d(8),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(2,1)))
        self.linblock = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(896,64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64,7)
        )        
    def forward(self,x):
        #set_trace()
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.linblock(x)
        return x
    
MODEL = CNN_classif()

def load_most_recent(model,model_dir):
    file_name = max([file  for root, dirs, files in os.walk(MODEL_DIR, topdown=False) for file in files])
    print("Loading {}".format(file_name))

    model.load_state_dict(torch.load(os.path.join(model_dir,file_name)))
    model.eval()

    
def train_model(model, inputs, targets,nb_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    batch_size = 20
    for e in range(nb_epochs):
        clear_output(wait=True)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        accuracy = (predicted == targets).sum().item() / inputs.shape[0] * 100
        print("Progression:{} % Accuracy: {:.2f}% ".format(e/nb_epochs*100,accuracy))
        for train_batch,target_batch in zip(inputs.split(batch_size),
                                targets.split(batch_size)):
            output_batch = model(train_batch)
            loss = criterion(output_batch,target_batch)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

def run_model(nb_epochs=5):
    """
    train and save the modle
    """
    file_names,data_f,targets = load_mfcc_data(MFCC_DIR)
    model = CNN_classif()
    data_f = torch.Tensor(data_f)
    targets = torch.Tensor(targets).long()
    train_model(model,data_f,targets.long(),nb_epochs)
    name = datetime.now().strftime("%m_%d_%H%M")
    torch.save(model.state_dict(), "./models/{}".format(name))
    torch.save(model.state_dict(), os.path.join(MODEL_DIR,name))

    return True


def dummy_model(filename):
    load_most_recent(MODEL,MODEL_DIR)
    data = wavfile.read(os.path.join(WAV_DIR,"{}.wav".format(filename)))[1]
    emotion = DE2EN[filename[5]]
    return {'true_label':FULL_EM[emotion],
            'predicted': FULL_EM[emotion]}

def smart_model(file_name):
    load_most_recent(MODEL,MODEL_DIR)
    target0 = FULL_EM[DE2EN[file_name[5]]]
    #load the corresonding mfcc file
    with open(os.path.join(MFCC_DIR,file_name + ".pkl"),'rb') as f:
        sample,target = pickle.load(f)
    sample = torch.Tensor(sample).unsqueeze(0)
    predicted = MODEL(sample).max(1)[1].item()
    #convert label to actual emotion
    predicted = FULL_EM[NUM2EN[predicted]]
    target = FULL_EM[NUM2EN[target]]
    assert(target==target0)
    return {'true_label':target,
            'predicted':predicted}

if __name__ == '__main__':
    #run_model(data_f,targets,nb_epochs=1)
    print(smart_model('03a04Fd'))