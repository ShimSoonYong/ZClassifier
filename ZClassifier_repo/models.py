# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import seaborn as sns
from numpy.linalg import norm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import bartlett
from scipy.stats import ttest_ind
import torchvision.models as models
from sklearn.decomposition import PCA
from numpy.linalg import norm, eigvals
import torchvision.transforms as transforms
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, Subset
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define ZClassifier with feature extractor as parameter
class ZClassifier(nn.Module):
    def __init__(self, feature_extractor, latent_dim=10, num_classes=10, beta=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.features = feature_extractor
        self.head = nn.Linear(self.features.out_dim, 128)
        self.mu_head = nn.Linear(128, num_classes)
        self.logvar_head = nn.Linear(128, num_classes)
        self.epsilon = Normal(0, beta)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(self.head(x))
        self.mu = self.mu_head(x).unsqueeze(-1).repeat(1, 1, self.latent_dim)
        self.logvar = self.logvar_head(x).unsqueeze(-1).repeat(1, 1, self.latent_dim)
        sigma = self.logvar.exp()
        eps = self.epsilon.sample((x.size(0), self.num_classes, self.latent_dim)).to(x.device)
        self.z = self.mu + sigma * eps
        return self.z.mean(-1)

# Define a standard CNN classifier (e.g., ResNet18) with softmax output
class SoftmaxClassifier(nn.Module):
    def __init__(self, feature_extractor, num_classes=10):
        super().__init__()
        self.features = feature_extractor
        self.head = nn.Linear(self.features.out_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)  # Logits (no softmax here; use later for scoring)
        return x

# Feature extractors
def load_model_with_weights(model_fn, variant, weights):
    if weights is None:
        return model_fn(weights=None)
    try:
        # torchvision >= 0.13 (uses Weights enums)
        weight_enum = getattr(models, f"{variant.upper()}_Weights")
        return model_fn(weights=weight_enum[weights])
    except AttributeError:
        # torchvision < 0.13 (uses pretrained=True)
        return model_fn(pretrained=(weights == 'IMAGENET1K_V1'))

class ResNetFeature(nn.Module):
    def __init__(self, variant='resnet18', weights='IMAGENET1K_V1'):
        super().__init__()
        assert variant in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        resnet = load_model_with_weights(getattr(models, variant), variant, weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class VGGFeature(nn.Module):
    def __init__(self, variant='vgg11', weights='IMAGENET1K_V1'):
        super().__init__()
        assert variant in ['vgg11', 'vgg13', 'vgg16', 'vgg19']
        vgg = load_model_with_weights(getattr(models, variant), variant, weights)
        self.features = vgg.features
        self.pool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.out_dim = vgg.classifier[0].in_features

    def forward(self, x):
        x = self.pool(self.features(x))
        x = self.flatten(x)
        return x
