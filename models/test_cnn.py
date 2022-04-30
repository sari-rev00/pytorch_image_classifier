from datetime import datetime
from pprint import pprint

import numpy as np
import os
import sys
import torch


os.chdir('..')
sys.path.append('.')

from models.cnn import (
    LW60, SuperLW60, ExstraSuperLW60, UltimateLW60, 
    MNIST, Inception60, Fire, SqueezedNet60)

if False:
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


def dimtester(t_in_dim, t_out_dim, model, model_name):
    input_tensor = torch.rand(t_in_dim)
    output_tensor = model(input_tensor)
    print("    {} :: output tensor size: {}, expected: {}".format(
        model_name,
        output_tensor.size(),
        torch.Size(t_out_dim)))
    assert output_tensor.size() == torch.Size(t_out_dim)


def test_SuperLW60():
    print("\n{} ==============".format(sys._getframe().f_code.co_name))
    class_num = 3

    d_params = {
        "dropout_feature": 0.3,
        "dropout_classifier": 0.3,
        "class_num": class_num}
    model = SuperLW60(d_params=d_params)
    print("    desc: {}".format(model.model_descriptions()))

    # features ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 16, 5, 5]
    dimtester(t_in_dim, t_out_dim, model.features, "features")

    # classifier ---------------------------------------------------------
    t_in_dim = [1, 16*5*5]
    t_out_dim = [1, class_num]
    dimtester(t_in_dim, t_out_dim, model.classifier, "classifier")

    # forwarding ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 3]
    dimtester(t_in_dim, t_out_dim, model, "forwarding")

    return None


def test_ExstraSuperLW60():
    print("\n{} ==============".format(sys._getframe().f_code.co_name))
    class_num = 3

    d_params = {
        "dropout_feature": 0.3,
        "dropout_classifier": 0.3,
        "class_num": class_num}
    model = ExstraSuperLW60(d_params=d_params)
    print("    desc: {}".format(model.model_descriptions()))

    # features ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 8, 5, 5]
    dimtester(t_in_dim, t_out_dim, model.features, "features")

    # classifier ---------------------------------------------------------
    t_in_dim = [1, 8*5*5]
    t_out_dim = [1, class_num]
    dimtester(t_in_dim, t_out_dim, model.classifier, "classifier")

    # forwarding ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 3]
    dimtester(t_in_dim, t_out_dim, model, "forwarding")

    return None


def test_UltimateLW60():
    print("\n{} ==============".format(sys._getframe().f_code.co_name))
    class_num = 3

    d_params = {
        "dropout_feature": 0.3,
        "dropout_classifier": 0.3,
        "class_num": class_num}
    model = UltimateLW60(d_params=d_params)
    print("    desc: {}".format(model.model_descriptions()))

    # features ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 8, 4, 4]
    dimtester(t_in_dim, t_out_dim, model.features, "features")

    # classifier ---------------------------------------------------------
    t_in_dim = [1, 8*4*4]
    t_out_dim = [1, class_num]
    dimtester(t_in_dim, t_out_dim, model.classifier, "classifier")

    # forwarding ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 3]
    dimtester(t_in_dim, t_out_dim, model, "forwarding")

    return None



def test_Fire():
    print("\n{} ==============".format(sys._getframe().f_code.co_name))
    f_dim = [64, 8, 64, 64]
    fire = Fire(
        in_ch=f_dim[0], 
        sq_ch=f_dim[1], 
        exp1x1_ch=f_dim[2], 
        exp3x3_ch=f_dim[3])
    
    t_in_dim = [1, 64, 60, 60]
    t_out_dim = [t_in_dim[0], f_dim[2] + f_dim[3], t_in_dim[2], t_in_dim[3]]

    dimtester(t_in_dim, t_out_dim, fire, "forwarding")
    return None


def test_SqueezedNet60():
    print("\n{} ==============".format(sys._getframe().f_code.co_name))
    class_num = 3

    d_params = {
        "dropout": 0.3,
        "class_num": class_num}
    sq = SqueezedNet60(d_params=d_params)

    print("    desc: {}".format(sq.model_descriptions()))
    
    # features ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 512, 5, 5]
    dimtester(t_in_dim, t_out_dim, sq.features, "features")

    # classifier ---------------------------------------------------------
    t_in_dim = [1, 512, 5, 5]
    t_out_dim = [1, class_num, 1, 1]
    dimtester(t_in_dim, t_out_dim, sq.classifier, "classifier")

    # forwarding ----------------------------------------------------------
    t_in_dim = [1, 3, 60, 60]
    t_out_dim = [1, 3]
    dimtester(t_in_dim, t_out_dim, sq, "forwarding")

    return None


if __name__ == '__main__':
    test_SuperLW60()
    test_ExstraSuperLW60()
    test_UltimateLW60()
    # test_Fire()
    # test_SqueezedNet60()
