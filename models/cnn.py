import os
import shutil
import torch
import torch.nn as nn
from pydantic import BaseModel, ValidationError, validator

from config.config import ConfManager


SAVE_DIR_BASE = ConfManager.SAVE_DIR_BASE


class CnnModelBase(nn.Module):
    '''
    - inherit this class to build your cnn model class.
    - you need to define folowing method on your cnn model class.
      - prepare_cnn()
      - forward()
      - model_descriptions(): include model_state_dict, model_descriptions, label_idx_dict
    '''

    def __init__(self, validator, d_params=None, model_info_fname=None, init_weights=True):
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
        v_params = validator(**self.d_params)
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
    class ParamValidator(BaseModel):
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

    def __init__(self, d_params=None, model_info_fname=None, init_weights=True):
        super().__init__(
            validator=self.ParamValidator,
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
    class ParamValidator(BaseModel):
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

    def __init__(self, d_params=None, model_info_fname=None, init_weights=True):
        super().__init__(
            validator=self.ParamValidator,
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


class Inception60(CnnModelBase):
    class ParamValidator(BaseModel):
        dropout_basic_conv: float
        dropout_inception: float
        dropout_classifier: float
        class_num: int

        @validator('dropout_basic_conv', 'dropout_inception', 'dropout_classifier')
        def check_dropout_range(cls, v):
            if not (0 <= v < 1.0):
                raise ValueError("range error: {}".format(v))
            return v
        
        @validator('class_num')
        def check_class_num(cls, v):
            if not (2 <= v <=10):
                raise ValueError("range error: {}".format(v))
            return v

    def __init__(self, d_params=None, model_info_fname=None, init_weights=True):
        super().__init__(
            validator=self.ParamValidator,
            d_params=d_params, 
            model_info_fname=model_info_fname, 
            init_weights=init_weights)
        return None
    
    def prepare_cnn(self, v_params):
        # in: 3x60x60
        self.basic_conv = nn.Sequential(            
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=v_params.dropout_basic_conv))
        # out: 64x15x15

        # in: 64x15x15
        self.inception_k1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(True))
        # out: 16x15x15

        # in: 64x15x15
        self.inception_k3 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(True))
        # out: 16x15x15

        # in: 64x15x15
        self.inception_k5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(True))
        # out: 32x15x15

        # in: 64x15x15
        self.inception_AP_k1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(True))
        # out: 16x15x15

        # in: 80x15x15
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3),
            nn.Dropout2d(p=v_params.dropout_inception))
        # in: 80x5x5(=2000)

        # in: 80x5x5(=2000)
        self.classifier = nn.Sequential(
            nn.Linear(80*5*5, 400, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=v_params.dropout_classifier),
            nn.Linear(400, 400, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=v_params.dropout_classifier),
            nn.Linear(400, v_params.class_num, bias=True))
        # out: 3
        
    def forward(self, x):
        x = self.basic_conv(x)
        out_k1 = self.inception_k1(x)
        out_k3 = self.inception_k3(x)
        out_k5 = self.inception_k5(x)
        out_AP_k1 = self.inception_AP_k1(x)
        out = self.pooling(
            torch.cat([out_k1, out_k3, out_k5, out_AP_k1], dim=1))
        out = out.view(-1, 80*5*5)
        out = self.classifier(out)
        return out
    
    def model_descriptions(self):
        return {
            "name": "Inception60",
            "input_size": int(60),
            "input_channel": int(3),
            "params": self.d_params}