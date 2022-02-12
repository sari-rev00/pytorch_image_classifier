from datetime import datetime
from pprint import pprint

import numpy as np
import os
import sys
import torch


os.chdir('..')
sys.path.append('.')

from models.cnn import LW60, MNIST, Inception60

if True:
    print("\nLW60")
    model_lm60 = LW60(d_params={
        "dropout_feature": 0.3,
        "dropout_classifier": 0.3,
        "class_num": 3})
    print("\nmodel_lm60: {}".format(model_lm60.model_descriptions()))
    print(model_lm60)

    # features =================================================
    print("\nfeatures")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 3 * 60 * 60).reshape([1, 3, 60, 60])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_lm60.features(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 64, 5, 5])

    # classifier =================================================
    print("\nclassifier")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 64 * 5 * 5).reshape([1, 64 * 5 * 5])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_lm60.classifier(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 3])

    # forward =================================================
    print("\nforward")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 3 * 60 * 60).reshape([1, 3, 60, 60])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_lm60.forward(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 3])


if False:
    model_mnist = MNIST(d_params={
        "dropout_feature": 0.3,
        "dropout_classifier": 0.3,
        "class_num": 10})
    print("\nmodel_mnist: {}".format(model_mnist.model_descriptions()))


    fname = "MNIST_20220103015715_016.pth"
    model_mnist_2 = MNIST(model_info_fname=fname)
    print("\nmodel_mnist_2: {}".format(model_mnist_2.model_descriptions()))
    print(model_mnist_2.label_idx_dict)

    dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
    model_mnist_2.save_model_info(fname=f"MNIST_{dt_now}_test")


if False:
    print("\nInception60")
    model_Inception60 = Inception60(d_params={
        "dropout_basic_conv": 0.3,
        "dropout_inception": 0.3,
        "dropout_classifier": 0.3,
        "class_num": 3})
    print("\nmodel_mnist: {}".format(model_Inception60.model_descriptions()))
    # print(model_Inception60)

    # basic_conv =================================================
    print("\nbasic_conv")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 3 * 60 * 60).reshape([1, 3, 60, 60])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_Inception60.basic_conv(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 64, 15, 15])

    # inception_k1 =================================================
    print("\ninception_k1")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 64 * 15 * 15).reshape([1, 64, 15, 15])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_Inception60.inception_k1(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 16, 15, 15])

    # inception_k3 =================================================
    print("\ninception_k3")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 64 * 15 * 15).reshape([1, 64, 15, 15])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_Inception60.inception_k3(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 16, 15, 15])

    # inception_k5 =================================================
    print("\ninception_k5")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 64 * 15 * 15).reshape([1, 64, 15, 15])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_Inception60.inception_k5(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 32, 15, 15])

    # inception_AP_k1 =================================================
    print("\ninception_AP_k1")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 64 * 15 * 15).reshape([1, 64, 15, 15])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_Inception60.inception_AP_k1(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 16, 15, 15])

    # classifier =================================================
    print("\nclassifier")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 80 * 5 * 5).reshape([1, 80 * 5 * 5])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_Inception60.classifier(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 3])

    # forward =================================================
    print("\nforward")
    print("input:")
    test_tensor = torch.tensor(
        np.random.rand(1 * 3 * 60 * 60).reshape([1, 3, 60, 60])).float()
    print(f"    input data (image) shape: {test_tensor.shape}")
    print("output:")
    output_tensor = model_Inception60.forward(test_tensor)
    print(f"    output data (image) shape: {output_tensor.shape}")
    assert output_tensor.shape == torch.Size([1, 3])