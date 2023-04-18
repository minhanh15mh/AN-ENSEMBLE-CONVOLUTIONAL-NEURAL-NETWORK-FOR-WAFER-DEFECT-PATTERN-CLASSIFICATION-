import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def make_datapath_list(directory):
  data = []
  label = []
  for folder in os.listdir(directory):
    for file in os.listdir(os.path.join(directory, folder)):
        file_path = os.path.join(directory, folder, file)
        data.append(file_path)
        label.append(folder)    
  return data, label

class ImageDataset(Dataset):
    
    def __init__(self, file_list, label_list):
        # the paths of images
        self.file_list = file_list

        # convert label from string to one hot encoding
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(label_list)
        # new_integer_encoded = tf.keras.utils.to_categorical(integer_encoded)

        self.label = integer_encoded

        # convert image from array to tensor form (from (H,W,C)-> (C,H,W))
        self.data_transform = transforms.Compose([transforms.ToTensor()])
 
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.file_list[i]))
        image = self.data_transform(image)
        label = self.label[i]
        return image, label

class SettingConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])
