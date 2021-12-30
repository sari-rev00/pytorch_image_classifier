import os
import sys
from pprint import pprint
import numpy as np

os.chdir('..')
sys.path.append('.')

# sys.path.append('utils')
# sys.path.append('test.test_imgs')

# print(os.getcwd())
# pprint(sys.path)

from utils.dataloader import DataFile, Dataset, DataTransform, DataLoader
from config.config import TransformParam, ConfDataloader
from test_imgs.test_data_01 import get_test_datainfo


list_dir = [
    "./test/test_imgs/img/bike/",
    "./test/test_imgs/img/car/"]
list_label = [
    1,
    0]

data_file = DataFile(dirs=list_dir, labels=list_label, split=True, test_size=0.2)
f_train, f_test, l_train, l_test = data_file.data()
pprint(f_train)
pprint(l_train)
pprint(f_test)
pprint(l_test)


data_transform = DataTransform(transform_param=TransformParam)

# fpaths, labels = get_test_datainfo()
# print(type(fpaths))
# seed = 12345
# np.random.seed(seed=seed)
# np.random.shuffle(fpaths)
# np.random.seed(seed=seed)
# np.random.shuffle(labels)
# print(type(fpaths))

# pprint(fpaths)
# pprint(labels)
# for f, l in zip(fpaths, labels):
#     print(f"{f}: {l}")



dataset = Dataset(
    x_train=f_train, 
    y_train=l_train, 
    x_test=f_test, 
    y_test=l_test, 
    transform=data_transform)


# dataset.set_mode(mode="train")
# print(dataset.__len__())
# print(dataset.__getitem__(index=0))

# dataset.set_mode(mode="test")
# print(dataset.__len__())
# print(dataset.__getitem__(index=0))


# train / test dataset -------------------------------
for mode in ["train", "test"]:
    print(f"mode: {mode} --------")
    dataset.set_mode(mode=mode)
    for idx in range(dataset.__len__()):
        x, y = dataset.__getitem__(index=idx)
        print(f"{type(x)}: {y}")


# dataloader -----------------------------------------
dataloader = DataLoader(dataset=dataset)

dataloader.set_mode("train")

for mode in ["train", "test"]:
    print("mode: {} ------------".format(mode))
    print("data num: {}".format(len(dataloader.set_mode(mode).dataset)))
    for x, y in dataloader.set_mode(mode):
        print(len(x))
        print(y)

for mode in ["train", "test"]:
    print(dataloader.set_mode(mode).data_num())

print(dataloader.data_num_all())


