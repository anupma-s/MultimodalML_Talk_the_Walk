import os
import sys
import shutil
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from mvae_model import Multimodal_AE


def compute_loss(recon_image, image, recon_lang, lang, mu, logvar,
              lambda_image=1.0, lambda_lang=1.0, annealing_factor=1):
    
    image_bce = 0
    lang_bce = 0
    
    image_bce = torch.sum(binary_cross_entropy_with_logits(
            recon_image.view(-1, 3 * 64 * 64), 
            image.view(-1, 3 * 64 * 64)), dim=1)

    lang_bce = torch.sum(binary_cross_entropy_with_logits(
            recon_lang, lang))


    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    loss = torch.mean(lambda_image * image_bce + lambda_lang * lang_bce 
                      + annealing_factor * kld)
    return loss

def binary_cross_entropy_with_logits(input, target):
    return (torch.clamp(input, 0) - input * target 
            + torch.log(1 + torch.exp(-torch.abs(input))))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=100,
                        help='size of the latent embedding [default: 100]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # crop images
	preprocess_data = transforms.Compose([transforms.Resize(64),
	                                          transforms.CenterCrop(64),
	                                          transforms.ToTensor()])
	# Load train data
	train_loader   = torch.utils.data.DataLoader()
	mb_size = len(train_loader)

	# Initialize model and Optimizer
	model     = Multimodal_AE(args.n_latents)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	if args.cuda:
        model.cuda()


    def train(epoch):
        model.train()
        total_train_loss = 0
        for batch_idx, (image, lang) in enumerate(train_loader):

            if args.cuda:
                image     = image.cuda()
                attrs     = attrs.cuda()

            image      = Variable(image)
            attrs      = Variable(attrs)
            batch_size = len(image)

            optimizer.zero_grad()
            recon_image, recon_lang, mu_1, logvar_1 = model(image, lang)
            train_loss = compute_loss(recon_image, image, recon_lang, lang, mu_1, logvar_1, 
                                   lambda_image=args.lambda_image, lambda_attrs=args.lambda_attrs)
            total_train_loss += train_loss
            total_avg_train_loss = total_train_loss/(batch_size*(batch_idx+1))
            train_loss.backward()
            optimizer.step()



            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), total_train_loss))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, total_train_loss))
