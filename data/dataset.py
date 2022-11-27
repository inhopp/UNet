import os
import csv
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class Dataset(data.Dataset):
    def __init__(self, opt, phase, transform=None):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.phase = phase
        self.transform = transform

        self.data = list()
        with open(os.path.join(self.data_dir, self.data_name, '{}.csv'.format(phase))) as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)

    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.data_name, self.phase, self.data[index][0])
        mask_img_path = os.path.join(self.data_dir, self.data_name, self.phase + '_mask', self.data[index][0])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_img_path).convert("L"), dtype=np.float32)
        mask[mask==255.0] = 1.0

        if self.transform is not None:
            image = self.transform(image)
            image = transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0))(image)
            mask = self.transform(mask)
            mask = transforms.Normalize((0.0), (1.0))(mask)

        return image, mask

    def __len__(self):
        return len(self.data)
