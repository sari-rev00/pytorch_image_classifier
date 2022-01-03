import os
import numpy as np
from PIL import Image
from pprint import pprint
from torchvision import datasets, transforms
import shutil


BASE_DIR = "./mnist_data/"
IMG_DIR = "imgs/"

def download_mnist_image_files(data_num, save_dir=BASE_DIR):
    transform = transforms.Compose([transforms.ToTensor()])
    data = datasets.MNIST(save_dir, download=True, transform=transform)
    print("preparing img files...")
    if os.path.exists(BASE_DIR + IMG_DIR):
        shutil.rmtree(BASE_DIR + IMG_DIR)
        print("removed stored img")
    os.mkdir(BASE_DIR + IMG_DIR)
    d_count = dict()
    for i in range(data_num):
        x, y = data.__getitem__(i)
        save_dir = BASE_DIR + IMG_DIR + str(y) + "/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        arr_x = np.squeeze(x.numpy().transpose((1, 2, 0)))
        im = Image.fromarray(arr_x * 255).convert("L")
        if not str(y) in d_count.keys():
            d_count[str(y)] = int(0)
        d_count[str(y)] += int(1)
        im.save(save_dir + "{}_{}.jpg".format(y, str(d_count[str(y)]).zfill(4)))
    print("Completed")
    return d_count