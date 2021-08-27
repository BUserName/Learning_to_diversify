import os
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn as nn

from utils.util import *


class AlexNetCaffe(nn.Module):
    def __init__(self, n_classes=100, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout())]))

        self.classifier_l = nn.Linear(512, n_classes)
        self.p_logvar = nn.Sequential(nn.Linear(4096, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(4096, 512),
                                  nn.LeakyReLU())

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.classifier_l.parameters(), self.p_logvar.parameters(), self.p_mu.parameters()
                                 ), "lr": base_lr}]

    def is_patch_based(self):
        return False

    def forward(self, x, train=True):
        end_points={}
        x = self.features(x*57.6)  #57.6 is the magic number needed to bring torch data back to the range of caffe data, based on used std
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        logvar = self.p_logvar(x)
        mu = self.p_mu(x)
        end_points['logvar'] = logvar
        end_points['mu'] = mu

        if train:
            x = reparametrize(mu, logvar)
        else:
            x = mu

        end_points['Embedding'] = x
        x = self.classifier_l(x)
        end_points['Predictions'] = nn.functional.softmax(input=x, dim=-1)


        return x, end_points


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def caffenet(classes):
    model = AlexNetCaffe(classes)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)

    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)

    return model

# class AlexNetCaffeAvgPool(AlexNetCaffe):
#     def __init__(self, jigsaw_classes=1000, n_classes=100):
#         super().__init__()
#         print("Global Average Pool variant")
#         self.features = nn.Sequential(OrderedDict([
#             ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
#             ("relu1", nn.ReLU(inplace=True)),
#             ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
#             ("relu2", nn.ReLU(inplace=True)),
#             ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
#             ("relu3", nn.ReLU(inplace=True)),
#             ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
#             ("relu4", nn.ReLU(inplace=True)),
#             ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
#             #             ("relu5", nn.ReLU(inplace=True)),
#             #             ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#         ]))
#         self.classifier = nn.Sequential(
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True))
#
#         self.jigsaw_classifier = nn.Sequential(
#             nn.Conv2d(1024, 128, kernel_size=3, stride=2, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             Flatten(),
#             nn.Linear(128 * 6 * 6, jigsaw_classes)
#         )
#         self.class_classifier = nn.Sequential(
#             nn.Conv2d(1024, n_classes, kernel_size=3, padding=1, bias=False),
#             nn.AvgPool2d(13),
#             Flatten(),
#             # nn.Linear(1024, n_classes)
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0.)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)



# class AlexNetCaffeFC7(AlexNetCaffe):
#     def __init__(self, jigsaw_classes=1000, n_classes=100, dropout=True):
#         super(AlexNetCaffeFC7, self).__init__()
#         print("FC7 branching variant")
#         self.features = nn.Sequential(OrderedDict([
#             ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
#             ("relu1", nn.ReLU(inplace=True)),
#             ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
#             ("relu2", nn.ReLU(inplace=True)),
#             ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
#             ("relu3", nn.ReLU(inplace=True)),
#             ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
#             ("relu4", nn.ReLU(inplace=True)),
#             ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
#             ("relu5", nn.ReLU(inplace=True)),
#             ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#         ]))
#         self.classifier = nn.Sequential(OrderedDict([
#             ("fc6", nn.Linear(256 * 6 * 6, 4096)),
#             ("relu6", nn.ReLU(inplace=True)),
#             ("drop6", nn.Dropout() if dropout else Id())]))
#
#         self.jigsaw_classifier = nn.Sequential(OrderedDict([
#             ("fc7", nn.Linear(4096, 4096)),
#             ("relu7", nn.ReLU(inplace=True)),
#             ("drop7", nn.Dropout()),
#             ("fc8", nn.Linear(4096, jigsaw_classes))]))
#         self.class_classifier = nn.Sequential(OrderedDict([
#             ("fc7", nn.Linear(4096, 4096)),
#             ("relu7", nn.ReLU(inplace=True)),
#             ("drop7", nn.Dropout()),
#             ("fc8", nn.Linear(4096, n_classes))]))





# def caffenet_gap(jigsaw_classes, classes):
#     model = AlexNetCaffe(jigsaw_classes, classes)
#     state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
#     del state_dict["classifier.fc6.weight"]
#     del state_dict["classifier.fc6.bias"]
#     del state_dict["classifier.fc7.weight"]
#     del state_dict["classifier.fc7.bias"]
#     del state_dict["classifier.fc8.weight"]
#     del state_dict["classifier.fc8.bias"]
#     model.load_state_dict(state_dict, strict=False)
#     # weights are initialized in the constructor
#     return model
#
#
# def caffenet_fc7(jigsaw_classes, classes):
#     model = AlexNetCaffeFC7(jigsaw_classes, classes)
#     state_dict = torch.load("models/pretrained/alexnet_caffe.pth.tar")
#     state_dict["jigsaw_classifier.fc7.weight"] = state_dict["classifier.fc7.weight"]
#     state_dict["jigsaw_classifier.fc7.bias"] = state_dict["classifier.fc7.bias"]
#     state_dict["class_classifier.fc7.weight"] = state_dict["classifier.fc7.weight"]
#     state_dict["class_classifier.fc7.bias"] = state_dict["classifier.fc7.bias"]
#     del state_dict["classifier.fc8.weight"]
#     del state_dict["classifier.fc8.bias"]
#     del state_dict["classifier.fc7.weight"]
#     del state_dict["classifier.fc7.bias"]
#     model.load_state_dict(state_dict, strict=False)
#     nn.init.xavier_uniform_(model.jigsaw_classifier.fc8.weight, .1)
#     nn.init.constant_(model.jigsaw_classifier.fc8.bias, 0.)
#     nn.init.xavier_uniform_(model.class_classifier.fc8.weight, .1)
#     nn.init.constant_(model.class_classifier.fc8.bias, 0.)
#     return model
