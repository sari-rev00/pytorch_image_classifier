import os
import shutil
import torch
import torch.nn as nn
from pydantic import BaseModel, ValidationError, validator

from config.config import ConfManager


SAVE_DIR_BASE = ConfManager.SAVE_DIR_BASE

class ValidatorBase(BaseModel):
    dropout_feature: float
    dropout_classifier: float
    class_num: int

    @validator('dropout_feature', 'dropout_classifier')
    def check_dropout_range(cls, v):
        if not (0 <= v < 1.0):
            raise ValueError("range error: {}".format(v))
        return v
    
    @validator('class_num')
    def check_class_num(cls, v):
        if not (2 <= v <=10):
            raise ValueError("range error: {}".format(v))
        return v


class CnnModelBase(nn.Module):
    '''
    - inherit this class to build your cnn model class.
    - you need to define folowing method on your cnn model class.
      - prepare_cnn()
      - forward()
      - model_descriptions(): include model_state_dict, model_descriptions, label_idx_dict
    '''
    class Validator(ValidatorBase):
        pass

    def __init__(self, d_params=None, model_info_fname=None, init_weights=True):
        if d_params:
            self.d_params = d_params
            self.label_idx_dict = None
        elif model_info_fname:
            if not ".pth" in model_info_fname:
                model_info_fname += ".pth"
            check_point = torch.load(SAVE_DIR_BASE + model_info_fname)
            self.d_params = check_point["model_descriptions"]["params"]
            self.label_idx_dict = check_point["label_idx_dict"]
        else:
            raise Exception("parameter error: one of d_params, model_info_fname is required")
        v_params = self.Validator(**self.d_params)
        super().__init__()
        self.prepare_cnn(v_params=v_params)
        if (not model_info_fname) and init_weights:
            self._initialize_weight()
        if (not d_params) and model_info_fname:
            self.load_state_dict(check_point["model_state_dict"])
        return None
    
    def prepare_cnn(self, v_params):
        return None
    
    def save_model_info(self, dir, fname):
        dict_info = {
            "model_state_dict": self.state_dict(),
            "model_descriptions": self.model_descriptions(),
            "label_idx_dict": self.label_idx_dict}
        if dir[-1] != "/":
            dir += "/"
        if not os.path.exists(SAVE_DIR_BASE + dir):
            os.mkdir(SAVE_DIR_BASE + dir)
        if not ".pth" in fname:
            fname += ".pth"
        torch.save(dict_info, SAVE_DIR_BASE + dir + fname)
        return None
    
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        return None


class MNIST(CnnModelBase):
    def __init__(self, d_params=None, model_info_fname=None, init_weights=True):
        super().__init__(
            d_params=d_params, 
            model_info_fname=model_info_fname, 
            init_weights=init_weights)
        return None
    
    def prepare_cnn(self, v_params):
        self.features = nn.Sequential(
            # input: 1x28x28
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=v_params.dropout_feature))
            # input: 32x7x7
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7, 32*7*7, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=v_params.dropout_classifier),
            nn.Linear(32*7*7, 512, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=v_params.dropout_classifier),
            nn.Linear(512, v_params.class_num, bias=True))
        return None
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32*7*7)
        x = self.classifier(x)
        return x
    
    def model_descriptions(self):
        return {
            "name": "MNIST",
            "input_size": int(28),
            "input_channel": int(1),
            "params": self.d_params}


class LW60(CnnModelBase):
    def __init__(self, d_params=None, model_info_fname=None, init_weights=True):
        super().__init__(
            d_params=d_params, 
            model_info_fname=model_info_fname, 
            init_weights=init_weights)
        return None
    
    def prepare_cnn(self, v_params):
        self.features = nn.Sequential(
            # input: 3x60x60
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3),
            nn.Dropout2d(p=v_params.dropout_feature))
            # output: 64x5x5
        self.classifier = nn.Sequential(
            nn.Linear(64*5*5, 64*5*5, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=v_params.dropout_classifier),
            nn.Linear(64*5*5, 64*5*5, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=v_params.dropout_classifier),
            nn.Linear(64*5*5, v_params.class_num, bias=True))
        return None
     
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64*5*5)
        x = self.classifier(x)
        return x
    
    def model_descriptions(self):
        return {
            "name": "LW60",
            "input_size": int(60),
            "input_channel": int(3),
            "params": self.d_params}
