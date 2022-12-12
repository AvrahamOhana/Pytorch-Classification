import glob
import os

import cv2
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, images_filepaths, classes, transform=None):

        self.images_filepaths = images_filepaths
        self.transform = transform
        self.classes = classes
        

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        folder_name = os.path.normpath(image_filepath).split(os.sep)[-2]
        label = float(self.classes.index(folder_name))

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label