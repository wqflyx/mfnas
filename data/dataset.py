import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np
import torch
import random
import pickle
import os


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class DataEncap:
    def __init__(self, opt):
        self.picture_size = opt.picture_size
        self.cutout = opt.cutout
        self.cutout_length = opt.cutout_length
        self.dataset_root = opt.dataset_root
        self.dataset_name = opt.dataset_name
        self.train_portion = opt.train_portion
        self.batch_size = opt.batch_size
        self.worker_nums = opt.worker_nums

    def data_generate(self, is_search=True):

        if self.dataset_name == "MNIST" or self.dataset_name == "FashionMNIST":
            train_transform, valid_transform = self._data_transforms_mnist()
            train_data = eval("dset." + self.dataset_name)(root=self.dataset_root, train=True, download=True,
                                                           transform=train_transform)
            test_data = eval("dset." + self.dataset_name)(root=self.dataset_root, train=False, download=True,
                                                          transform=valid_transform)
        elif self.dataset_name == "CIFAR10":
            train_transform, valid_transform = self._data_transforms_cifar10()
            train_data = eval("dset." + self.dataset_name)(root=self.dataset_root, train=True, download=True,
                                                           transform=train_transform)
            test_data = eval("dset." + self.dataset_name)(root=self.dataset_root, train=False, download=True,
                                                          transform=valid_transform)

        elif self.dataset_name == "SVHN" or self.dataset_name == "STL10":
            train_transform, valid_transform = self._data_transforms_cifar10()
            train_data = eval("dset." + self.dataset_name)(root=self.dataset_root, download=True,
                                                           transform=train_transform)
            test_data = eval("dset." + self.dataset_name)(root=self.dataset_root, split='test', download=True,
                                                          transform=valid_transform)

        elif self.dataset_name == "ImageFolder":
            resize_value = 0
            for i in range(20):
                if np.power(2, i) >= self.picture_size:
                    resize_value = int(np.power(2, i))
                    break
            train_transform, valid_transform = self._data_transforms_imagenet(resize_value)
            train_data = dset.ImageFolder(root=self.dataset_root, transform=train_transform)
            test_data = dset.ImageFolder(root=self.dataset_root + "_val", transform=valid_transform)
        else:
            print("not supported dataset!!!!")
            return []

        num_train = len(train_data)
        indices = list(range(num_train))

        if self.dataset_name == "ImageFolder":  # shuffle data to avoid imagefolder load problem
            np.random.shuffle(indices)

        split = int(np.floor(self.train_portion * num_train))

        if is_search:
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                pin_memory=True, num_workers=self.worker_nums)

            valid_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                pin_memory=True, num_workers=self.worker_nums)

            return train_queue, valid_queue

        else:
            print("test dataset loading..................")
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.worker_nums)

            test_queue = torch.utils.data.DataLoader(
                test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.worker_nums)

            return train_queue, test_queue

    def data_generate_raw(self):

        if self.dataset_name == "ImageFolder":
            resize_value = 0
            for i in range(20):
                if np.power(2, i) >= self.picture_size:
                    resize_value = int(np.power(2, i))
                    break
            train_transform, _ = self._data_transforms_imagenet(resize_value)
            train_data = dset.ImageFolder(root=self.dataset_root, transform=train_transform)
        else:
            print("not supported dataset!!!!")
            return []

        if os.path.exists("./model/imagenet_idx.txt"):
            print("loading imagent idx!!!!!!!")
            filename = "./model/imagenet_idx.txt"
            fr = open(filename, "rb")
            idx = pickle.load(fr)
            fr.close()
        else:
            print("generating imagent idx!!!!!!!")
            idx = []
            for class_name in train_data.classes:
                # get the indices in the dataset that are relative to that class
                idx.append([pos for pos, item in enumerate(train_data.samples)
                            if item[1] == train_data.class_to_idx[class_name]])

            filename = "./model/imagenet_idx.txt"
            fw = open(filename, 'wb')
            pickle.dump(idx, fw)
            fw.close()

        return train_data, idx

    def data_generate_imagenet_balance_sampler(self, train_data, idx, samples_per_class):

        indices = []
        for i in range(len(idx)):
            indices += random.sample(idx[i], samples_per_class)

        print("samples:::::::,  indices:::", samples_per_class, indices[0], indices[100])

        queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory=True, num_workers=self.worker_nums)

        return queue

    def _data_transforms_mnist(self):

        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if self.cutout:
            train_transform.transforms.append(Cutout(self.cutout_length))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return train_transform, valid_transform

    def _data_transforms_cifar10(self):
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(self.picture_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        if self.cutout:
            train_transform.transforms.append(Cutout(self.cutout_length))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        return train_transform, valid_transform

    def _data_transforms_imagenet(self, resize_value):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.picture_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(resize_value),
            transforms.CenterCrop(self.picture_size),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, valid_transform
