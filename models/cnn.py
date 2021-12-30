import torch.nn as nn
from pydantic import BaseModel, ValidationError, validator


class Validator(BaseModel):
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


class LW60(nn.Module):
    def __init__(self, d_params, init_weights=True):
        v_params = Validator(**d_params)
        super().__init__()
        self.d_params = d_params
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
        if init_weights:
            self._initialize_weights()
        return None
     
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64*5*5)
        x = self.classifier(x)
        return x
    
    def model_descriptions(self):
        return {
            "name": "LW60",
            "params": self.d_params}
    
    def _initialize_weights(self):
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