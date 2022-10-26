import os
import random
import re
from pathlib import Path
import json
import cv2, zlib, base64, io
import io

import torchvision
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2
import torch
from skimage.io import imread
from torch.utils import data
import torchvision.transforms.functional as F


class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None, device='cpu'
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.device = device
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.normalize = dict(
            mean= [0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406])


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        black_mark = False
        try:
            input_ID = self.inputs[index]
            target_ID = self.targets[index]
        except:
            index = random.randint(0, len(self.targets)-1)
            input_ID = self.inputs[index]
            target_ID = self.targets[index]
        # Load input and target

        x = cv2.imread(input_ID)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        try:
            y = imread(target_ID)
        except:
            y = cv2.imread(target_ID)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

        if 'detectron' in target_ID:
            y = (y==2).astype(float)
            if np.mean(y) < 0.1:
                y = np.ones((3,512,512))*255
                del self.inputs[index]
                del self.targets[index]
                return self.__getitem__(index)

            # else:
            #     plt.imshow(x)
            #     plt.show()
            #     plt.imshow(y)
            #     plt.show()


        x = torch.Tensor(cv2.resize(x, (512, 512))).float() / 255
        y = torch.Tensor(cv2.resize(y, (512, 512))).float() / y.max()
        #
        if len(y.shape) == 3:
            y = y.permute((2, 0, 1))
            y = y[0,:,:]
            y = (y > 0.05).to(dtype=torch.float)

        if len(x.shape) == 2:
            x = x.unsqueeze(2)
            x = torch.concat([x, x, x], dim=2)
            # x = x.concat([x, x, x], dim=2)

        if x.shape[2] == 4:
            x = x[:, :, :3]
        try:
            x = x.permute((2, 0, 1))
        except:
            print('X shape issue ')
            print(x.shape)

        x = F.normalize(x, self.normalize["mean"], self.normalize["std"])
        y = y.unsqueeze(0)

        x, y = x.cuda().to(self.device), y.cuda().to(self.device)
        # print('seg data y min max', y.max(), y.min())
        return x, y

    @staticmethod
    def base64_2_mask(s):
        z = zlib.decompress(base64.b64decode(s))
        n = np.fromstring(z, np.uint8)
        mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
        return mask

    @staticmethod
    def mask_2_base64(mask):
        img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
        img_pil.putpalette([0, 0, 0, 255, 255, 255])
        bytes_io = io.BytesIO()
        img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
        bytes = bytes_io.getvalue()
        return base64.b64encode(zlib.compress(bytes)).decode('utf-8')

    def imread_supervisely_json(self, path):
        with open(path) as file:
            data = json.load(file)
        if 'points' in data['objects'][0]:
            mask = np.array(data['objects'][0]['points']['exterior'])
            filled = np.zeros((data['size']['height'], data['size']['width']))
            filled = cv2.fillPoly(filled, pts=[mask], color=(255, 255, 255))
            mask = np.array(filled)
            if bool(data['objects'][0]['points']['interior']):
                mask1 = np.array(data['objects'][0]['points']['interior'][0])
                filled = np.zeros((data['size']['height'], data['size']['width']))
                filled = cv2.fillPoly(filled, pts=[mask1], color=(255, 255, 255))
                # plt.imshow(mask)
                # plt.show()
                # plt.imshow(mask1)
                # plt.show()
                mask = mask + np.array(filled)
                mask = np.clip(mask, 0, 255)

        elif 'bitmap' in data['objects'][0]:
            bmp_data = data['objects'][0]['bitmap']['data']
            mask_small = self.base64_2_mask(bmp_data)
            mask = np.zeros((data['size']['height'], data['size']['width']))
            start_point = data['objects'][0]['bitmap']['origin']
            mask[
            start_point[1]:start_point[1] + mask_small.shape[0],
            start_point[0]:start_point[0] + mask_small.shape[1]
            ] = mask_small
            mask *= 255
        return mask

    def imread_similars_json(self, path, shape):
        with open(path) as file:
            data = json.load(file)
        mask = np.array(data['shapes'][0]['points'], dtype=np.int32)
        filled = np.zeros((shape[0], shape[1]))
        filled = cv2.fillPoly(filled, pts=[mask], color=(255, 255, 255))
        return np.array(filled)


def construct_dataset(data_dirs, device):
    inputs = []
    targets = []
    for name, data_dir in data_dirs.items():
        # if 'supervisely' in name:
        #     for ds in list(Path(data_dir).iterdir()):
        #         if '.' in str(ds):
        #             continue
        #         inputs += sort(list(Path(os.path.join(ds, 'img')).iterdir()))
        #         targets += sort(list(Path(os.path.join(ds, 'ann')).iterdir()))
        # elif 'crowd_instance' in name:
        #     inputs += sort(list(Path(os.path.join(data_dir, 'Images')).iterdir()))
        #     targets += sort(list(Path(os.path.join(data_dir, 'Human')).iterdir()))
        # else:
        #     curr_inputs = sort(list(Path(os.path.join(data_dir, 'img')).iterdir()))
        #     curr_targets = sort(list(Path(os.path.join(data_dir, 'ann')).iterdir()))
        #     if 'fashion' in name:
        #         curr_inputs = curr_inputs[-200:]
        #         curr_targets = curr_targets[-200:]
        #     inputs += curr_inputs
        #     targets += curr_targets
        if 'images' in name:
            inputs += sort(list(Path(data_dir).iterdir()))
        if 'masks' in name:
            targets += sort(list(Path(data_dir).iterdir()))

    dataset = SegmentationDataSet(inputs=inputs,
                                  targets=targets,
                                  transform=None, device=device)
    return dataset


def get_dataloaders(training_dataset, val_dataset, configs):
    training_dataloader = data.DataLoader(dataset=training_dataset,
                                          batch_size=configs['batch_size'],
                                          shuffle=True)

    val_dataloader = data.DataLoader(dataset=val_dataset,
                                     batch_size=1,
                                     shuffle=False)

    return {'train': training_dataloader, 'val': val_dataloader}, {'train': len(training_dataset),
                                                                   'val': len(val_dataset)}


def sort(list_of_path_obj):
    list_of_strs = [str(p) for p in list_of_path_obj]
    list_of_strs = sorted(list_of_strs)
    list_of_strs = sorted(
        list_of_strs, key=lambda x: list(map(int, re.findall(r"\d+", x)))[-1]
    )
    return list_of_strs
