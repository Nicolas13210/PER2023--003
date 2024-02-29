from torch.utils.data import Dataset, DataLoader
from config import cfg
import os
import random
import matplotlib.pyplot as plt
from matplotlib import image as Img
import torch
import numpy as np
import torchvision
import json


class image_dataset(Dataset):
    '''
    this class is made to load your data, preprocess them and give them to your dataloader
    data_paths: list which contains the path of all data you want to manage with this dataset
    train: Boolean representing if your dataset is a training dataset or not
    return: None
    '''
    def __init__(self, data_paths, train, command_list):
        self.data_paths = data_paths #list of (d, instant, image, depth)
        self.train_mode = train
        self.command_lines = []
        for file_path in command_list:
            with open(file_path, 'r') as f:
                self.command_lines.append(f.readlines())
        
        self.mean = -0.11460872207730237
        self.std = 0.2924843364744082

    '''
    this function is made to give you the data at the index idx
    idx: integer representing the index of the data you want to get
    return:
        x_data: a tensor which contains your data
        labels: a tensor which contains the corresponding labels
    '''
    def __getitem__(self, idx):
        image = torch.tensor(Img.imread(self.data_paths[idx][2])[120:])
        image = torch.swapaxes(image, -1, 0)
        image = torch.swapaxes(image, -1, 1)

        depth = torch.tensor(Img.imread(self.data_paths[idx][3]))
        depth = torch.unsqueeze(depth, dim=0)

        command_dict = json.loads(self.command_lines[self.data_paths[idx][0]][self.data_paths[idx][1]])

        direction = (torch.tensor(command_dict['steering_angle']) - torch.tensor(self.mean))/self.std
        speed = torch.tensor(command_dict['speed']/cfg.DATASET.MAX_SPEED)

        # return x_data.float(), torch.nn.functional.one_hot(torch.tensor(self.labels[idx]), len(cfg.DATASET.LABELS)).float()
        return image.float(), depth.float(), speed.float(), direction.float()
    
    
    '''
    this function is made to return your dataset's len
    args:
        None
    
    return: integer which represent the len of your dataset
    '''
    def __len__(self):
        return len(self.data_paths)
    

'''
this function is made to build train, validation and test dataloader
args:
    none
return:
    train_dataloader: instance of dataloader which deal with your training data
    val_dataloader: instance of dataloader which deal with your validation data
    test_dataloader: instance of dataloader which deal with your test data
'''
def get_dataloader():

    data_list = []
    command_list = []
    for d,data_file in enumerate(cfg.DATASET.USED_DATA_FOLDER):
        command_list.append(os.path.join(data_file, 'commands.json'))

        for instant in range(len(os.listdir(os.path.join(data_file,'images')))):
            topics_list = []
            topics_list.append(d)
            topics_list.append(instant)
            topics_list.append(os.path.join(data_file, os.path.join('images', f'image_instant_{instant}.jpeg')))
            topics_list.append(os.path.join(data_file, os.path.join('depths', f'depth_0_instant_{instant}.jpeg')))

            data_list.append(topics_list)

    random.Random(4).shuffle(data_list)

    

    train_dataloader = DataLoader(image_dataset(data_list[:int(cfg.DATASET.TRAIN_PROPORTION*len(data_list))], True, command_list), batch_size=cfg.DATASET.TRAINING_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(image_dataset(data_list[int(cfg.DATASET.TRAIN_PROPORTION*len(data_list)):], False, command_list), batch_size=cfg.DATASET.VALIDATION_BATCH_SIZE, shuffle=True)
    
    return train_dataloader, val_dataloader

class image_classif_dataset(Dataset):
    '''
    this class is made to load your data, preprocess them and give them to your dataloader
    data_paths: list which contains the path of all data you want to manage with this dataset
    train: Boolean representing if your dataset is a training dataset or not
    return: None
    '''
    def __init__(self, data_paths, train, command_list):
        self.data_paths = data_paths #list of (d, instant, image, depth)
        self.train_mode = train
        self.command_lines = []
        for file_path in command_list:
            with open(file_path, 'r') as f:
                self.command_lines.append(f.readlines())
        

    '''
    this function is made to give you the data at the index idx
    idx: integer representing the index of the data you want to get
    return:
        x_data: a tensor which contains your data
        labels: a tensor which contains the corresponding labels
    '''
    def __getitem__(self, idx):
        image = torch.tensor(Img.imread(self.data_paths[idx][2])[120:])
        image = torch.swapaxes(image, -1, 0)
        image = torch.swapaxes(image, -1, 1)

        command_dict = json.loads(self.command_lines[self.data_paths[idx][0]][self.data_paths[idx][1]])

        if command_dict['steering_angle'] > 0.6:
            direction = 0
        elif command_dict['steering_angle'] > 0.4:
            direction = 1
        elif command_dict['steering_angle'] > 0.2:
            direction = 2
        elif command_dict['steering_angle'] > 0:
            direction = 3
        elif command_dict['steering_angle'] > -0.2:
            direction = 4
        elif command_dict['steering_angle'] > -0.4:
            direction = 5
        elif command_dict['steering_angle'] > -0.6:
            direction = 6
        else:
            direction = 7

        # direction = torch.nn.functional.one_hot(torch.tensor(direction), 8)
        direction = torch.tensor(direction)

        return image.float(), direction #.float()
    
    
    '''
    this function is made to return your dataset's len
    args:
        None
    
    return: integer which represent the len of your dataset
    '''
    def __len__(self):
        return len(self.data_paths)
    

'''
this function is made to build train, validation and test dataloader
args:
    none
return:
    train_dataloader: instance of dataloader which deal with your training data
    val_dataloader: instance of dataloader which deal with your validation data
    test_dataloader: instance of dataloader which deal with your test data
'''
def get_classif_dataloader():

    data_list = []
    command_list = []
    for d,data_file in enumerate(cfg.DATASET.USED_DATA_FOLDER):
        command_list.append(os.path.join(data_file, 'commands.json'))

        for instant in range(len(os.listdir(os.path.join(data_file,'images')))):
            topics_list = []
            topics_list.append(d)
            topics_list.append(instant)
            topics_list.append(os.path.join(data_file, os.path.join('images', f'image_instant_{instant}.jpeg')))

            data_list.append(topics_list)

    random.Random(4).shuffle(data_list)

    train_dataloader = DataLoader(image_classif_dataset(data_list[:int(cfg.DATASET.TRAIN_PROPORTION*len(data_list))], True, command_list), batch_size=cfg.DATASET.TRAINING_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(image_classif_dataset(data_list[int(cfg.DATASET.TRAIN_PROPORTION*len(data_list)):], False, command_list), batch_size=cfg.DATASET.VALIDATION_BATCH_SIZE, shuffle=True)
    
    return train_dataloader, val_dataloader

class image_dataset_LSTM(Dataset):
    '''
    this class is made to load your data, preprocess them and give them to your dataloader
    data_paths: list which contains the path of all data you want to manage with this dataset
    train: Boolean representing if your dataset is a training dataset or not
    return: None
    '''
    def __init__(self, data_paths, train, command_list):
        self.data_paths = data_paths #list of (d, instant, image, depth)
        self.train_mode = train
        self.command_lines = []
        for file_path in command_list:
            with open(file_path, 'r') as f:
                self.command_lines.append(f.readlines())

    '''
    this function is made to give you the data at the index idx
    idx: integer representing the index of the data you want to get
    return:
        x_data: a tensor which contains your data
        labels: a tensor which contains the corresponding labels
    '''
    def __getitem__(self, idx):

        sequence = torch.rand((cfg.TRAIN.SEQUENCE_SIZE, cfg.TRAIN.IMAGE_SHAPE[0], cfg.TRAIN.IMAGE_SHAPE[1], cfg.TRAIN.IMAGE_SHAPE[2]))
        label_seq = torch.randint(0,1,(cfg.TRAIN.SEQUENCE_SIZE,))
        for i in range(cfg.TRAIN.SEQUENCE_SIZE):

            image = torch.tensor(Img.imread(self.data_paths[idx + i][2])[120:])
            image = torch.swapaxes(image, -1, 0)
            image = torch.swapaxes(image, -1, 1).float()
            sequence[i] = image

            direction = json.loads(self.command_lines[self.data_paths[idx + i][0]][self.data_paths[idx + i][1]])['steering_angle']

            if direction > 0.6:
                label = 0
            elif direction > 0.4:
                label = 1
            elif direction > 0.2:
                label = 2
            elif direction > 0.0:
                label = 3
            elif direction > -0.2:
                label = 4
            elif direction > -0.4:
                label = 5
            elif direction > -0.6:
                label = 6
            else:
                label = 7

            label_seq[i] = label
        # return x_data.float(), torch.nn.functional.one_hot(torch.tensor(self.labels[idx]), len(cfg.DATASET.LABELS)).float()
        return sequence.float(), label_seq
    
    
    '''
    this function is made to return your dataset's len
    args:
        None
    
    return: integer which represent the len of your dataset
    '''
    def __len__(self):
        return len(self.data_paths) - cfg.TRAIN.SEQUENCE_SIZE
    

'''
this function is made to build train, validation and test dataloader
args:
    none
return:
    train_dataloader: instance of dataloader which deal with your training data
    val_dataloader: instance of dataloader which deal with your validation data
    test_dataloader: instance of dataloader which deal with your test data
'''
def get_dataloader_LSTM():

    data_list = []
    command_list = []
    for d,data_file in enumerate(cfg.DATASET.USED_DATA_FOLDER):
        command_list.append(os.path.join(data_file, 'commands.json'))

        for instant in range(len(os.listdir(os.path.join(data_file,'images')))):
            topics_list = []
            topics_list.append(d)
            topics_list.append(instant)
            topics_list.append(os.path.join(data_file, os.path.join('images', f'image_instant_{instant}.jpeg')))

            data_list.append(topics_list)

    random.Random(4).shuffle(data_list)

    

    train_dataloader = DataLoader(image_dataset_LSTM(data_list[:int(cfg.DATASET.TRAIN_PROPORTION*len(data_list))], True, command_list), batch_size=cfg.DATASET.TRAINING_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(image_dataset_LSTM(data_list[int(cfg.DATASET.TRAIN_PROPORTION*len(data_list)):], False, command_list), batch_size=cfg.DATASET.VALIDATION_BATCH_SIZE, shuffle=True)
    
    return train_dataloader, val_dataloader