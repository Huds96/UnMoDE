import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import copy
import yaml
import importlib

def gazeto2d(gaze):
    yaw = -np.arctan2(-gaze[0], -gaze[2])
    pitch = -np.arcsin(-gaze[1])
    return np.array([yaw, pitch])

def Decode_IVOrigin(line):
    anno = edict()
    anno.face = line[0]
    anno.name = line[0]
    anno.gaze = line[1]
    anno.placeholder = line[2]
    anno.zone = line[3]
    # anno.target = line[4]
    anno.origin = line[5]
    return anno

def Decode_IVNorm(line):
    anno = edict()
    anno.t1_face = line[8]
    anno.t1_gaze = line[9]
    
    anno.face = line[0]
    anno.gaze = line[1]
    
    anno.name = line[0]
    anno.head = line[2]
    anno.zone = line[3]
    anno.origin = line[4]
    anno.norm = line[6]
    return anno
  
def Decode_LBW(line):
    anno = edict()   
    anno.face = line[0]
    anno.gaze = line[3]
    anno.name = line[0]
    return anno


def Decode_Dict():
    mapping = edict()
    mapping.ivorigin = Decode_IVOrigin
    mapping.ivnorm = Decode_IVNorm
    mapping.lbw = Decode_LBW
    return mapping

def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)

def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

class commonloader(Dataset): 

  def __init__(self, dataset):

    self.source = edict() 
    self.source.norm = edict()
    norm = dataset.norm

    self.source.norm.root = norm.image 
    self.source.norm.line = self.__readlines(norm.label, norm.header)
    self.source.norm.decode = Get_Decode(norm.name)
    self.transforms = transforms.Compose([
        transforms.ToTensor()
    ])


  def __readlines(self, filename, header=True):

    data = []
    if isinstance(filename, list):
      for i in filename:
        with open(i) as f: line = f.readlines()
        if header: line.pop(0)
        data.extend(line)

    else:
      with open(filename) as f: data = f.readlines()
      if header: data.pop(0)
    return data


  def __len__(self):
    return len(self.source.norm.line) 


  def __getitem__(self, idx):

    # --------------------read norm------------------------
    line = self.source.norm.line[idx]
    line = line.strip().split(" ")

    # decode the info
    anno = self.source.norm.decode(line)

    # read image
    norm_img = cv2.imread(os.path.join(self.source.norm.root, anno.face))
    norm_img = self.transforms(norm_img)

    # read label
    norm_label = gazeto2d(np.array(anno.gaze.split(",")).astype("float"))
    norm_label = torch.from_numpy(norm_label).type(torch.FloatTensor)
    
    # ---------------------------------------------------
    data = edict()
    data.norm_face = norm_img
    data.name = anno.name

    label = edict()
    label.normGaze = norm_label
    
    return data, label


def loader(source, batch_size, shuffle=False,  num_workers=0):

  dataset = commonloader(source)

  print(f"-- [Read Data]: Total num: {len(dataset)}")

  print(f"-- [Read Data]: Source: {source.norm.label}")

  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load

