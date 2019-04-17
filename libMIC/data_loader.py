from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class PhrosisDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string or list): Path to the csv file(s) with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if(type(csv_file) is list):
            frame_list = []
            for one_csv in csv_file:
                frame_list.append(pd.read_csv(one_csv))
            self.items_class = pd.concat(frame_list, axis=0, ignore_index=True)
        else:
            self.items_class = pd.read_csv(csv_file)
        
        self.root_dir   = root_dir
        self.transform  = transform

    def get_csv_header(self):
        return self.items_class.columns.values
        
    def __len__(self):
        return len(self.items_class)

    def __getitem__(self, idx):
        img_name = self.items_class.iloc[idx, 0]
        img_full_name = "{0:}/{1:}".format(self.root_dir, img_name)
        image = np.asarray(io.imread(img_full_name), np.float32)
        label = self.items_class.iloc[idx, 1:].as_matrix()
        label = np.asarray(label.reshape([9,1]), np.int32)
        sample = {'image': image, 'label': label, 'name':img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        
        sample['image'] = img
        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        sample['image'] = image
        return sample

class Normalize(object):
    def __init__(self, mean ,std):
        self.mean = mean
        self.std  = std

    def __call__(self, sample):
        image = sample['image']
        image = (image - self.mean)/self.std

        sample['image'] = image
        return sample

class RandomNoise(object):
    def __init__(self, mean ,std):
        self.mean = mean
        self.std  = std

    def __call__(self, sample):
        image = sample['image']
        noise = np.random.normal(self.mean, self.std, size = image.shape)
        image = image + noise
        sample['image'] = image
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # print('label ', label)
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'name': sample['name']}

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = \
            sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    img = grid.numpy().transpose((1, 2, 0))
    img = img * 10 + 200
    img = np.asarray(img, np.uint8)
    plt.imshow(img)

if __name__ == "__main__":
    csv_files = ['D:/BaiduNetdiskDownload/phrosis_data/names/new_fold_1.csv',
                 'D:/BaiduNetdiskDownload/phrosis_data/names/new_fold_2.csv',
                 'D:/BaiduNetdiskDownload/phrosis_data/names/new_fold_3.csv',
                 'D:/BaiduNetdiskDownload/phrosis_data/names/new_fold_4.csv']
    root_dir = 'D:/BaiduNetdiskDownload/phrosis_data'
    transformed_dataset = PhrosisDataset(csv_file=csv_files,
                                        root_dir=root_dir,
                                        transform=transforms.Compose([
                                                RandomCrop((180, 230)),
                                                Rescale((192, 192)),
                                                Normalize(mean=[240.18, 238.28, 245.64],
                                                std=[16.74, 21.02, 4.59]),
                                                ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    # Helper function to show a batch


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['label'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break