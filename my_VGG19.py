from torchvision import models
from torch import nn


def VGG19_6way(pretrained: bool):
    model_ft = models.vgg19(pretrained=pretrained)
    num_ftrs = model_ft.classifier[6].in_features
    features = list(model_ft.classifier.children())[:-1]
    features.extend([nn.Linear(num_ftrs, 6)])
    model_ft.classifier = nn.Sequential(*features)
    return model_ft
