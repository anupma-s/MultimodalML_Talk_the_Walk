import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F



class Multimodal_AE(nn.Module):

    def __init__(self, n_latents, training=True):
        super(Multimodal_AE, self).__init__()
        self.img_enc = ImageEncoder(n_latents)
        self.n_latents     = n_latents
        self.training = training

    def forward(self, image=None, attrs=None):
        # mu, logvar  = self.infer(image, attrs)
        batch_size = image.size(0) if image is not None else attrs.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA

        # mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                  # use_cuda=use_cuda)
        mu     = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        

        if image is not None:
            image_mu, image_logvar = self.image_encoder(image)
            mu     = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if attrs is not None:
            attrs_mu, attrs_logvar = self.lang_encoder(attrs)
            mu     = torch.cat((mu, attrs_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, attrs_logvar.unsqueeze(0)), dim=0)

        # mu, logvar = self.experts(mu, logvar)
        eps=1e-8
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        mu, logvar = pd_mu, pd_logvar
        
        
        # z           = self.reparametrize(mu, logvar)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z =  eps.mul(std).add_(mu)
        else:
            z = mu
        

        image_recon = self.image_decoder(z)
        lang_recon = self.attrs_decoder(z)
        return image_recon, lang_recon, mu, logvar



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
        self.upsample = nn.Sequential(
            nn.Linear(n_latents, 256 * 5 * 5),
            Swish())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False))

    def forward(self, z):
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.deconv(z)
        return z  

class LanguageEncoder(nn.Module):
    def __init__(self, n_latents):
        super(LanguageEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_ATTRS, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]

class LanguageDecoder(nn.Module):
    def __init__(self, n_latents):
        super(LanguageDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, N_ATTRS))

    def forward(self, z):
        z = self.net(z)
        return z  

class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)



