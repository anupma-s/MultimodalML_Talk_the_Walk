import numpy as np
import torch

from tqdm import tqdm
import argparse
import datetime
import os
import pickle

from data_dialog import TextDataset, TalkTheWalk, collate_fn
from model_dialog import *


parser = argparse.ArgumentParser()
# parser.add_argument("--log", action='store_true', default=True)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--val", type=int, default=5)
# parser.add_argument("--name", type=str, default='task4')
parser.add_argument("-b", type=int, default=8, dest='batch_size')
# parser.add_argument("--cuda", action='store_true', default=False)
# parser.add_argument("--load_from", type=str, default=None)
# parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')
args = parser.parse_args()


ttw = TalkTheWalk(args.data_dir,'train')
train, w2i = list(ttw.read_dataset())
ttw = TalkTheWalk(args.data_dir,'valid')
val, _ = list(ttw.read_dataset())
# ttw = TalkTheWalk(args.data_dir,'test')
# test, _ = list(ttw.read_dataset())

data_train = TextDataset(train, w2i)
data_dev = TextDataset(val, w2i)
# data_test = TextDataset(test, w2i)

i2w = {v: k for k, v in w2i.items()}

train_data_loader = torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                                pin_memory=True,
                                                num_workers=2)

dev_data_loader = torch.utils.data.DataLoader(dataset=data_dev,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              pin_memory=True,
                                              num_workers=2)
#
# test_data_loader = torch.utils.data.DataLoader(dataset=data_test,
#                                               batch_size=args.batch_size,
#                                               shuffle=False,
#                                               collate_fn=collate_fn,
#                                                pin_memory=True,
#                                                num_workers=2)

model = Model(3, len(w2i), 128, 128, w2i)
for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for i, batch in pbar:
            model.train()
            loss = model.train_batch(batch[0].transpose(0, 1), batch[1].transpose(0, 1), i == 0, batch[2], batch[3], args.batch_size)
            pbar.set_description(model.print_loss())
        # if epoch+1 % 2 == 0:
        pbar2 = tqdm(enumerate(dev_data_loader), total=len(dev_data_loader))
        for j, data_dev in pbar2:
            words = model.evaluate_batch(len(data_dev[1]), data_dev[0].transpose(0, 1), data_dev[3], data_dev[6])
            print(words)
