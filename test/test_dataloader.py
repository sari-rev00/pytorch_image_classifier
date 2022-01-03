import os
import sys
from pprint import pprint
import numpy as np

os.chdir('..')
sys.path.append('.')

from utils.dataloader import DataFile, Dataset, DataTransform, DataLoader
from config.config import TransformParam, ConfDataloader
from test_imgs.test_data_01 import get_test_datainfo


list_dir = [
    "./test/test_imgs/img/bike/",
    "./test/test_imgs/img/car/"]
list_label = [
    "bike",
    "car"]

data_file = DataFile(dirs=list_dir, labels=list_label, split=True, test_size=0.2)
f_train, f_test, l_train, l_test = data_file.data()
uq_labels = data_file.unique_labels()
pprint(f_train)
pprint(l_train)
pprint(f_test)
pprint(l_test)
print(uq_labels)


data_transform = DataTransform(transform_param=TransformParam)

dataset = Dataset(
    x_train=f_train, 
    y_train=l_train, 
    x_test=f_test, 
    y_test=l_test, 
    labels=uq_labels,
    transform=data_transform)


# train / test dataset -------------------------------
for mode in ["train", "test"]:
    print(f"mode: {mode} --------")
    dataset.set_mode(mode=mode)
    for idx in range(dataset.__len__()):
        x, y = dataset.__getitem__(index=idx)
        print(f"{type(x)}: label_idx {y}, label {dataset.label_idx2label(label_index=y.item())}")


# dataloader -----------------------------------------
dataloader = DataLoader(dataset=dataset)

dataloader.set_mode("train")

for mode in ["train", "test"]:
    print("mode: {} ------------".format(mode))
    print("data num: {}".format(len(dataloader.set_mode(mode).dataset)))
    for xs, ys in dataloader.set_mode(mode):
        print(len(xs))
        print([(dataloader.dataset.label_idx2label(y), y) for y in ys.tolist()])

for mode in ["train", "test"]:
    print(dataloader.set_mode(mode).data_num())

print(dataloader.data_num_all())


