import os
import sys
from pprint import pprint
from datetime import datetime

os.chdir('..')
sys.path.append('.')

import torch.optim as optim
import torch.nn as nn
from utils.model_manager import Manager
from utils.dataloader import DataFile, Dataset, DataTransform, DataLoader, gen_dataloader
from config.config import TransformParam, ConfDataloader
from models.cnn import LW60


list_dir = [
    "./test/test_imgs/img/bike/",
    "./test/test_imgs/img/car/"]
list_label = [
    "bike",
    "car"]

# dataloader -----------------------------------
# data_file = DataFile(dirs=list_dir, labels=list_label, split=True, test_size=0.2)
# f_train, f_test, l_train, l_test = data_file.data()
# data_transform = DataTransform(transform_param=TransformParam)
# dataset = Dataset(
#     x_train=f_train, 
#     y_train=l_train, 
#     x_test=f_test, 
#     y_test=l_test, 
#     transform=data_transform)
# dataloader = DataLoader(dataset=dataset)

dataloader, data_descriptions = gen_dataloader(
    data_dirs=list_dir, 
    labels=list_label, 
    split=True, 
    test_size=0.2)

print(dataloader.dataset.label_idx_dict)
print(data_descriptions)
class_num = data_descriptions["class_num"]
# -----------------------------------------------

if True:
    model = LW60(
        d_params={
            "dropout_feature": 0.3,
            "dropout_classifier": 0.3,
            "class_num": class_num})

    optimizer = optim.SGD(
        params=model.parameters(), 
        lr=0.0001, 
        momentum=0.9,
        weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    mm = Manager(model=model)
    mm.train(
        num_epochs=30, 
        dataloader=dataloader, 
        auto_save=True,
        print_epoch_step=int(10))

    pprint(mm.training_result)

    mm.save_model_info(dir="test/", fname="{}_test_{}".format(
        model.model_descriptions()["name"],
        datetime.now().strftime('%Y%m%d%H%M%S')))

if False:
    mm_2 = Manager(
        model=LW60(d_params={
            "dropout_feature": 0.3,
            "dropout_classifier": 0.3,
            "class_num": class_num}))

    mm_2.load_model_info(fname="test/LW60_test_20220103125553.pth")

    pred_label = mm_2.predict("./test/test_imgs/img/bike/bike_01.jpg", pos=False)
    print("pred_label: {}".format(pred_label))

    pred_pos = mm_2.predict("./test/test_imgs/img/bike/bike_01.jpg", pos=True)
    print("pred_pos: {}".format(pred_pos))
    print(mm_2.model.label_idx_dict)

    