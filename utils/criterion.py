import torch.nn as nn


def default_criterion():
    return nn.CrossEntropyLoss()