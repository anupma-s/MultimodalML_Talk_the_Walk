import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F



class Multimodal_AE(nn.Module):

    def __init__(self, n_latents):
        super(Multimodal_AE, self).__init__()
        self.img_enc = ImageEncoder(n_latents)
        self.n_latents     = n_latents

    def forward(self, image=None, attrs=None):
        return



class ImageEncoder(nn.Module):
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]

class ImageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()

class LanguageEncoder(nn.Module):
    def __init__(self, n_latents):
        super(LanguageEncoder, self).__init__()

class LanguageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(LanguageDecoder, self).__init__()

class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)



