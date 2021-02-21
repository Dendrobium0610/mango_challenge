from torch.utils.data import Dataset
from utils import loadTxt
from PIL import Image
import numpy as np
import torch
import random
import os
import pandas as pd


def default_loader(path):
    return Image.open(path)

class ImageDataSet(Dataset):
	def __init__(self, csv_path, img_dir, loader=default_loader, transform=False, sort=False):
		self.csv = pd.read_csv(csv_path)    
		fileNames = list(self.csv["filename"].values)
		self.classes =list(self.csv.columns.values[1:])
		if sort:
			fileNames.sort()
		self.file_names = fileNames
		self.img_dir = img_dir
		self.loader = loader
		self.transform = transform

	def preprocess(self, img, size, label=False):
		img = img.resize(size, Image.BILINEAR)
		if self.transform:
			img = self.transform(img)           
		img = np.array(img)
		img_trans = img.transpose((2, 0, 1))
		return img_trans / 255.0

	def __getitem__(self, idx):
		file_name = self.file_names[idx]
		label = self.csv.loc[self.csv["filename"]==file_name, self.classes].values
		img = self.loader(self.img_dir/file_name)
		label = np.array(label).reshape(-1)      
		img = self.preprocess(img, (256, 256))

		return torch.from_numpy(img), torch.from_numpy(label)

	def __len__(self):
		return len(self.file_names)