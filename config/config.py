class ConfDataloader():
    BATCH_SIZE = 8
    SHUFFLE = True
    DROP_LAST = True
    TARGET_EXT = [".jpg", ".jpeg"]
    RANDOM_SEED = 12345


class ConfManager():
    ACC_TH = 0.98
    SAVE_DIR_BASE = "./weight_data/"


class ConfOptimizer():
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-5


# class TransformParam():
#     resize = 28
#     color_mean = [0.5]
#     color_std = [0.5]


class TransformParam():
    resize = 60
    color_mean = [0.5, 0.5, 0.5]
    color_std = [0.5, 0.5, 0.5]