from datetime import datetime
from tqdm import tqdm
from PIL import Image
import torch
import torch.optim as optim
from torch.autograd import Variable

from utils.dataloader import gen_transform
from utils.optimizer import default_optimizer
from utils.criterion import default_criterion
from config.config import ConfManager


ACC_TH = ConfManager.ACC_TH
SAVE_DIR_BASE = ConfManager.SAVE_DIR_BASE

class Manager():
    def __init__(self, model, label_num_dict=None):
        self.model = model
        self.label_num_dict = label_num_dict
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
        if not optimizer:
            optimizer = default_optimizer(self.model)
        if not criterion:
            criterion = default_criterion()
        if not print_epoch_step:
            print_epoch_step = int(1)
        dt_start = datetime.now()
        model_desc = self.model.model_descriptions()
        result = {
            "start": dt_start.strftime('%Y-%m-%d %H:%M:%S'),
            "model_descriptions": model_desc,
            "scores": list()}
        if self.label_num_dict:
            result["label_num"] = self.label_num_dict
        best_loss = None
        for ep in range(num_epochs):
            if ((ep + 1) % print_epoch_step) == 0:
                print("Epoch:{}/{} ============".format(ep + 1, num_epochs))
            d_score = dict()
            for mode in ["train","test"]:
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
                ep_loss_per_data = round(ep_loss / dataloader.data_num(), 3)
                ep_acc = round(ep_corrects / dataloader.data_num(), 3)
                if mode == "train":
                    d_score["train_loss"] = ep_loss_per_data
                    d_score["train_acc"] = ep_acc
                else:
                    d_score["test_loss"] = ep_loss_per_data
                    d_score["test_acc"] = ep_acc
                    if not best_loss:
                        best_loss = ep_loss_per_data
                    elif (ep_acc > acc_th) and (ep_loss < ep_loss_per_data) and auto_save:
                        fname = "{}_{}_{}".format(
                            model_desc["name"], 
                            dt_start.strftime('%Y%m%d%H%M%S'),
                            str(ep).zfill(3))
                        self.save_weighte(fname=fname)
                if ((ep + 1) % print_epoch_step) == 0:
                    print("    Mode: {}, Loss: {}, Acc: {}".format(
                        mode, 
                        ep_loss_per_data, 
                        ep_acc))
            result["scores"].append(d_score)
        result["end"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.training_result = result
        return None
    
    def save_weight(self, fname):
        if not ".pth" in fname:
            fname += ".pth"
        torch.save(self.model.state_dict(), SAVE_DIR_BASE + fname)
        return None
    
    def load_weight(self, fname):
        if not ".pth" in fname:
            fname += ".pth"
        self.model.load_state_dict(torch.load(SAVE_DIR_BASE + fname))
        return None
    
    def predict(self, fpath, pos=False):
        img = Image.open(fpath)
        transform = gen_transform()
        x = transform(img, "test")
        x = torch.reshape(x, (-1, 3, 60, 60))
        self.model.eval()
        pred_pos = self.model(x)
        if pos:
            return pred_pos[0].tolist()
        else:
            pred_label = torch.max(pred_pos, 1).indices.item()
            if self.label_num_dict:
                for k, v in self.label_num_dict.items():
                    if int(v) == int(pred_label):
                        return str(k)
                raise Exception(f"Error: predicted label {pred_label} doesn't match labels in label_num_dict.")
            else:
                return pred_label
        

