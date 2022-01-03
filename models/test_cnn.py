import os
import sys
from pprint import pprint
from datetime import datetime

os.chdir('..')
sys.path.append('.')

from models.cnn import LW60, MNIST

model_lm60 = LW60(d_params={
    "dropout_feature": 0.3,
    "dropout_classifier": 0.3,
    "class_num": 3})
print("\nmodel_lm60: {}".format(model_lm60.model_descriptions()))


model_mnist = MNIST(d_params={
    "dropout_feature": 0.3,
    "dropout_classifier": 0.3,
    "class_num": 10})
print("\nmodel_mnist: {}".format(model_mnist.model_descriptions()))


fname = "MNIST_20220103015715_016.pth"
model_mnist_2 = MNIST(model_info_fname=fname)
print("\nmodel_mnist_2: {}".format(model_mnist_2.model_descriptions()))
print(model_mnist_2.label_idx_dict)

dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
model_mnist_2.save_model_info(fname=f"MNIST_{dt_now}_test")
