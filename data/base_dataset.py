import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class BaseDataset(torch.utils.data.Dataset):
    """docstring for BaseDataset"""
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return os.path.basename(self.opt.data_root.strip('/'))

    def initialize(self, opt):
        self.opt = opt
        self.imgs_dir = os.path.join(self.opt.data_root, self.opt.imgs_dir)
        self.is_train = self.opt.mode == "train"

        # load images path 
        filename = self.opt.train_list if self.is_train else self.opt.test_list
        self.imgs_name_file = os.path.join(self.opt.data_root, filename)
        self.imgs_path = self.make_dataset()

        # load AUs dicitionary 
        aus_pkl = os.path.join(self.opt.data_root, self.opt.aus_pkl + '_' + self.opt.control_signal_type + '.pkl')
        self.aus_dict = self.load_dict(aus_pkl)

        # load image to tensor transformer
        self.img2tensor = self.img_transformer()


    def make_dataset(self):
        return None

    def load_dict(self, pkl_path):
        saved_dict = {}
        with open(pkl_path, 'rb') as f:
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def get_aus_by_path(self, img_path):
        return None

    def img_transformer(self):
        transform_list = []
        transform_list.append(transforms.Resize([self.opt.load_size, self.opt.load_size], Image.BICUBIC))

        if self.is_train and self.opt.pad_and_crop:
          if self.opt.control_signal_type in  {'class','au'}:
            transform_list.append(transforms.Pad(padding=self.opt.final_size//20, padding_mode='reflect'))
            transform_list.append(transforms.RandomCrop(self.opt.final_size))

        transform_list.append(transforms.Resize([self.opt.final_size, self.opt.final_size], Image.BICUBIC))
        if self.is_train and not self.opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())


        if self.is_train and self.opt.random_adjust_color:
            transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        img2tensor = transforms.Compose(transform_list)

        return img2tensor

    def __len__(self):
        return len(self.imgs_path)





    







