import torch
import os
import numpy as np
import seaborn as sns
import time as t
import pandas as pd
from torch import nn
import torchaudio
import librosa
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader,Dataset

#####
class UTGAN_CustomDataset(Dataset):

    def __init__(self, dir,chunk,skip,label,transform=None):
        self.dir = dir
        self.chunk = chunk
        self.skip = skip
        self.label = label
        self.transform = transform
        self.subfile = sorted([x[0] for x in os.walk(self.dir)][1:])
        self.lenlist = [len([name for name in os.listdir(file)]) for file in self.subfile]
        self.lendict = {file : len( os.listdir(file)) for file in self.subfile}
        self.subdict = {file : sorted(os.listdir(file))[:self.lendict[file]] for file in self.subfile}
        self.data = []
        for folder, files in self.subdict.items():
            for file in self.subdict[folder]: 
                data = torchaudio.load(os.path.join(folder,file))[0][0]
                print('file: ',file,'len',len(data),'sampling rate: ',torchaudio.load(os.path.join(folder,file))[1])
                data = [data[i*self.skip:i*self.skip+self.chunk] for i in range((len(data)-self.chunk)//self.skip)]
                data = torch.vstack(data)
                self.data.append(data)
        self.data = torch.vstack(self.data)
        print('length of data',self.__len__())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output = self.data[idx]
        if self.transform :
            for i in self.transform:
                output = i(output)
        return output, self.label

#####
def save_model(path,model,genoptim,disoptim,latent,lr,alpha,los,name=""):
    torch.save({    
            'model_state_dict': model.state_dict(),
            'genoptimizer_state_dict': genoptim.state_dict(),
            'disoptimizer_state_dict': disoptim.state_dict(),
            'lr':lr,
            'alpha':alpha,
            'loss': los,
            }, path+f"//lat:{str(latent)};lr:{lr};alp:{alpha};name:"+name)
    print('saved')
    
#####
def load_model(path,model,genoptim,disoptim,latent,lr,alpha,los,name=""):
    model_checkpoint=torch.load(path+f"//lat:{str(latent)};lr:{lr};alp:{alpha};name:"+name)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    genoptim.load_state_dict(model_checkpoint['genoptimizer_state_dict'])
    disoptim.load_state_dict(model_checkpoint['disoptimizer_state_dict'])
    print('loaded')
    return model_checkpoint['loss']