import os
from datetime import datetime
import json
# from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.optim as optim
from torch.autograd import Variable

from utils.dataloader import gen_transform
from utils.optimizer import default_optimizer
from utils.criterion import default_criterion
from config.config import ConfManager, ConfOptimizer, TransformParam


ACC_TH = ConfManager.ACC_TH
SAVE_DIR_BASE = ConfManager.SAVE_DIR_BASE
DEFAULT_SAVE_DIR = "./result/"
COLOR_TRAIN = "deepskyblue"
COLOR_TEST = "tomato"
ROUND_DIGIT = 4

class Manager():
    def __init__(self, model):
        self.model = model
        self.training_result = None
        return None
    
    def train(
            self, 
            num_epochs, 
            dataloader, 
            optimizer=None, 
            criterion=None, 
            acc_th=ACC_TH,
            auto_save=True,
            print_epoch_step=None):
        self.model.label_idx_dict = dataloader.dataset.label_idx_dict
        if not optimizer:
            optimizer = default_optimizer(self.model)
        if not criterion:
            criterion = default_criterion()
        if not print_epoch_step:
            print_epoch_step = int(1)
        dt_start = datetime.now()
        model_desc = self.model.model_descriptions()
        if auto_save:
            save_dir = "{}_{}/".format(
                        model_desc["name"], 
                        dt_start.strftime('%Y%m%d%H%M%S'))
        result = {
            "start": dt_start.strftime('%Y-%m-%d %H:%M:%S'),
            "model_descriptions": model_desc,
            "scores": list()}
        result["label_idx_dict"] = self.model.label_idx_dict
        best_loss = None
        print("Training: {} {}\n".format(
            model_desc["name"], 
            dt_start.strftime('%Y%m%d%H%M%S')))
        for ep in range(1, num_epochs +1):
            if (ep % print_epoch_step) == 0:
                print("Epoch:{}/{} ============".format(ep, num_epochs))
            d_score = dict()
            d_score["epoch"] = ep
            for mode in ["train", "test"]:
                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()
                ep_loss = float(0)
                ep_corrects = int(0)
                dataloader.set_mode(mode=mode)
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(mode == "train"):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if mode == "train":
                            loss.backward()
                            optimizer.step()
                    ep_loss += loss.item() * inputs.size(0)
                    ep_corrects += torch.sum(preds == labels.data).item()
                ep_loss_per_data = round(ep_loss / dataloader.data_num(), ROUND_DIGIT)
                ep_acc = round(ep_corrects / dataloader.data_num(), ROUND_DIGIT)
                if mode == "train":
                    d_score["train_loss"] = ep_loss_per_data
                    d_score["train_acc"] = ep_acc
                else:
                    d_score["test_loss"] = ep_loss_per_data
                    d_score["test_acc"] = ep_acc
                    if not best_loss:
                        best_loss = ep_loss_per_data
                    elif (ep_acc > acc_th) and (ep_loss_per_data < best_loss) and auto_save:
                        best_loss = ep_loss_per_data
                        fname = "{}_{}_{}".format(
                            model_desc["name"], 
                            dt_start.strftime('%Y%m%d%H%M%S'),
                            str(ep).zfill(3))
                        self.save_model_info(dir=save_dir, fname=fname)
                if (ep % print_epoch_step) == 0:
                    print("    Mode: {}, Loss: {}, Acc: {}".format(
                        mode, 
                        ep_loss_per_data, 
                        ep_acc))
            result["scores"].append(d_score)
        result["end"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.training_result = result
        if auto_save:
            print("saved: training information")
            self.save_training_info(dir=save_dir, fname="training_info")
            self.make_result_fig(save=True, save_dir=SAVE_DIR_BASE + save_dir)
        return None
    
    def save_training_info(self, dir, fname):
        if not os.path.exists(SAVE_DIR_BASE + dir):
            os.mkdir(SAVE_DIR_BASE + dir)
        dict_info = {
            "training_result": self.training_result,
            "optimizer": {
                "learning_rate": ConfOptimizer.LEARNING_RATE,
                "momentum": ConfOptimizer.MOMENTUM,
                "weight_decay": ConfOptimizer.WEIGHT_DECAY},
            "transform": {
                "resize": TransformParam.resize,
                "color_mean": TransformParam.color_mean,
                "color_std": TransformParam.color_std}}
        if dir[-1] != "/":
            dir += "/"
        if not ".json" in fname:
            fname += ".json"
        with open(SAVE_DIR_BASE + dir + fname, mode='w') as f:
            json.dump(dict_info, f, indent=4)
        return None
    
    def save_model_info(self, dir, fname):
        self.model.save_model_info(dir=dir, fname=fname)
        return None
    
    def load_model_info(self, fname):
        self.model.__init__(model_info_fname=fname)
        return None
    
    def predict(self, fpath, pos=False):
        input_channel = self.model.model_descriptions()["input_channel"]
        input_size = self.model.model_descriptions()["input_size"]
        img = Image.open(fpath)
        transform = gen_transform()
        x = transform(img, "test")
        x = torch.reshape(x, (-1, input_channel, input_size, input_size))
        self.model.eval()
        pred_pos = self.model(x)
        if pos:
            return pred_pos[0].tolist()
        else:
            pred_label = torch.max(pred_pos, 1).indices.item()
            for k, v in self.model.label_idx_dict.items():
                if int(v) == int(pred_label):
                    return str(k)
            raise Exception(f"Error: predicted label {pred_label} is not included in label_idx_dict.")
        
    def make_result_fig(self, save=False, save_dir=DEFAULT_SAVE_DIR):
        if not self.training_result:
            return None
    
        color_train = COLOR_TRAIN
        color_test = COLOR_TEST

        df = pd.DataFrame(self.training_result["scores"])
        ep = df["epoch"].astype(int).values.tolist()
        train_loss = df["train_loss"].values.tolist()
        train_acc = df["train_acc"].values.tolist()
        test_loss = df["test_loss"].values.tolist()
        test_acc = df["test_acc"].values.tolist()
        model_name = self.model.model_descriptions()["name"]

        fig = plt.figure(figsize=(16, 6))
        ax_0 = fig.add_subplot(1,2,1)
        ax_0.plot(ep, train_loss, marker="o", markersize=6, color=color_train, label="train")
        ax_0.plot(ep, test_loss, marker="o", markersize=6, color=color_test, label="test")
        ax_0.set_yscale('log')
        ax_0.set_title('loss: ' + model_name + self.training_result["start"])
        ax_0.set_xlabel('epoch')
        ax_0.set_ylabel('loss')
        ax_0.grid(True)
        ax_0.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)

        ax_1 = fig.add_subplot(1,2,2)
        ax_1.plot(ep, train_acc, marker="o", markersize=6, color=color_train, label="train")
        ax_1.plot(ep, test_acc, marker="o", markersize=6, color=color_test, label="test")
        ax_1.set_ylim([0.95, 1.005])
        ax_1.set_title('acc: ' + model_name + self.training_result["start"])
        ax_1.set_xlabel('epoch')
        ax_1.set_ylabel('acc')
        ax_1.grid(True)
        ax_1.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=12)

        if save:
            dt = datetime.strptime(self.training_result["start"], '%Y-%m-%d %H:%M:%S')
            fig.savefig(save_dir + "{}_{}.jpg".format(
                self.training_result["model_descriptions"]["name"],
                dt.strftime('%Y%m%d%H%M%S')))
        return fig
        