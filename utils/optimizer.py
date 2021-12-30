import torch.optim as optim
from config.config import ConfOptimizer


LEARNING_RATE = ConfOptimizer.LEARNING_RATE
MOMENTUM = ConfOptimizer.MOMENTUM
WEIGHT_DECAY = ConfOptimizer.WEIGHT_DECAY

def default_optimizer(model):
    return optim.SGD(
        params=model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY)