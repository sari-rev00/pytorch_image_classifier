from cnn import LW60

d_param = {
    "dropout_feature": 0.3,
    "dropout_classifier": 0.3,
    "class_num": 3
}

model = LW60(d_params=d_param)

print(model.model_descriptions())