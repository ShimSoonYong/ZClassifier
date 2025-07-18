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
        self.mu = self.mu_head(x).unsqueeze(-1)
        self.logvar = self.logvar_head(x).unsqueeze(-1)
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

# Feature extractors for ResNet18 and VGG11
class ResNet18Feature(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.features(x).squeeze()
        return x

class VGG11Feature(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg11(weights=None)
        self.features = vgg.features
        self.pool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.out_dim = 512 * 7 * 7

    def forward(self, x):
        x = self.pool(self.features(x))
        x = self.flatten(x)
        return x