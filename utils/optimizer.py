import torch.optim as optim
from config.config import ConfOptimizer


LEARNING_RATE = ConfOptimizer.LEARNING_RATE
MOMENTUM = ConfOptimizer.MOMENTUM
WEIGHT_DECAY = ConfOptimizer.WEIGHT_DECAY

def default_optimizer(
        model, 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY):
    return optim.SGD(
        params=model.parameters(), 
        lr=lr, 
        momentum=momentum,
        weight_decay=weight_decay)