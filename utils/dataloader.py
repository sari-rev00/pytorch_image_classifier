import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

from config.config import ConfDataloader, TransformParam


BATCH_SIZE = ConfDataloader.BATCH_SIZE
SHUFFLE = ConfDataloader.SHUFFLE
DROP_LAST = ConfDataloader.DROP_LAST
TARGET_EXT = ConfDataloader.TARGET_EXT
RANDOM_SEED = ConfDataloader.RANDOM_SEED

class DataFile():
    def __init__(self, dirs, labels, shuffle=True, split=True, test_size=0.2):
        dirs = [str(d) for d in dirs]
        labels = [str(l) for l in labels]
        self.l_fpath, self.l_label = self.datafile_list(dirs=dirs, labels=labels)
        self.uq_labels = set(labels)
        self.shuffle = shuffle
        self.split = split
        self.test_size = test_size
        return None
    
    def unique_labels(self):
        return self.uq_labels
    
    def data_num_per_class(self):
        l, counts = np.unique(self.l_label, return_counts=True)
        return dict([(l, c) for l, c in zip(l, counts)])
    
    def datafile_list(self, dirs, labels):
        l_fpath = list()
        l_label = list()
        for dir, label in zip(dirs, labels):
            for fname in os.listdir(dir):
                if not os.path.splitext(fname)[-1] in TARGET_EXT:
                    continue
                l_fpath.append(os.path.abspath(os.path.join(dir, fname)))
                l_label.append(label)
        return l_fpath, l_label
    
    def shuffle_data(self):
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(self.l_fpath)
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(self.l_label)
        return None
    
    def data(self):
        if self.split:
            return train_test_split( # X_train, X_test, y_train, y_test
                self.l_fpath,
                self.l_label, 
                test_size=self.test_size, 
                random_state=RANDOM_SEED,
                stratify=self.l_label)
        else:
            if self.shuffle:
                self.shuffle_data()
            return self.l_fpath, self.l_label


class Dataset(data.Dataset):
    def __init__(self, x_train, y_train, x_test, y_test, labels, transform):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.transform = transform
        self.mode = "train"
        self.label_idx_dict, self.uq_labels = self.gen_label_idx_dict(labels=labels)
        return None
    
    def gen_label_idx_dict(self, labels):
        uq_labels = set(labels)
        label_idx_dict = dict([(str(label), i) for i, label in enumerate(uq_labels)])
        return label_idx_dict, uq_labels
    
    def data_num(self):
        d_count = dict()
        for tp in [("train", self.y_train), ("test", self.y_test)]:
            l, counts = np.unique(tp[1], return_counts=True)
            d_count[tp[0]] = dict([(l, c) for l, c in zip(l, counts)])
        return d_count
    
    def label_idx2label(self, label_index):
        for k, v in self.label_idx_dict.items():
            if int(v) == int(label_index):
                return k
        raise Exception(f"Label_index error: {label_index} is not included in label_idx_dict")
    
    def __len__(self):
        if self.mode == "train":
            return len(self.y_train)
        else:
            return len(self.y_test)
    
    def __getitem__(self, index):
        if self.mode == "train":
            img = Image.open(self.x_train[index])
            label = self.y_train[index]
        else:
            img = Image.open(self.x_test[index])
            label = self.y_test[index]
        x = self.transform(img, self.mode)
        y = torch.tensor(self.label_idx_dict[label], dtype=torch.long)
        return x, y
    
    def get_filename_label(self, index):
        if self.mode == "train":
            fn = self.x_train[index]
            l_idx = self.y_train[index]
        else:
            fn = self.x_test[index]
            l_idx = self.y_test[index]
        return fn, l_idx
    
    def get_mode(self):
        return self.mode
    
    def set_mode(self, mode):
        if mode == "train":
            self.mode = "train"
        else:
            self.mode = "test"
        return self


class DataTransform():
    def __init__(self, transform_param, preprocess=None):
        base_process = [
            transforms.Resize((transform_param.resize, transform_param.resize)),
            transforms.ToTensor(),
            transforms.Normalize(transform_param.color_mean, transform_param.color_std)
        ]
        self.data_transform = {
            "train": transforms.Compose(base_process),
            "test": transforms.Compose(base_process)
        }
        self.preprocess = preprocess
        return None
    
    def __call__(self, img, mode="train"):
        if self.preprocess:
            img = self.preprocess(img)
        return self.data_transform[mode](img)


class DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=DROP_LAST):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        super().__init__(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last)
        return None
    
    def set_mode(self, mode):
        if mode == "test":
            self.dataset.set_mode(mode="test")
        else:
            self.dataset.set_mode(mode="train")
        return self

    def get_mode(self):
        return self.dataset.get_mode()
    
    def data_num(self):
        return self.dataset.__len__()
    
    def data_num_all(self):
        return {
            "train": self.dataset.set_mode("train").__len__(), 
            "test": self.dataset.set_mode("test").__len__()}


def gen_dataloader(data_dirs, labels, split=True, test_size=0.2):
    data_file = DataFile(dirs=data_dirs, labels=labels, split=split, test_size=test_size)
    f_train, f_test, l_train, l_test = data_file.data()
    uq_labels = data_file.unique_labels()
    data_transform = DataTransform(transform_param=TransformParam)
    dataset = Dataset(
        x_train=f_train, 
        y_train=l_train, 
        x_test=f_test, 
        y_test=l_test,
        labels=uq_labels,
        transform=data_transform)
    deta_descriptions = {
        "class": uq_labels,
        "class_num": len(uq_labels),
        "data_num": dataset.data_num()
    }
    return DataLoader(dataset=dataset), deta_descriptions


def gen_transform():
    return DataTransform(transform_param=TransformParam)
