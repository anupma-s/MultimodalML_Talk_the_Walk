import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pdb
import json
import os

use_cuda = torch.cuda.is_available()

class TextDataset(Dataset):
    def __init__(self, memory, w2i):
        self.memory = memory
        self.w2i = w2i
        self.preprocess()

    def preprocess(self):
        """ performs word to index conversion for every element
                """
        for idx in range(len(self.memory)):
            context_seq = self.memory[idx][0]
            bot_seq = self.memory[idx][1]
            new_context_seq = []
            for c in context_seq:
                l = []
                for word in c:
                    l.append(self.w2i[word])
                new_context_seq.append(l)

            new_bot_seq = []
            for word in bot_seq.split(' '):
                new_bot_seq.append(self.w2i[word])
            new_bot_seq.append(self.w2i['<eos>'])
            self.memory[idx].append(new_context_seq)
            self.memory[idx].append(new_bot_seq)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx][3], self.memory[idx][4], self.memory[idx][2], self.memory[idx][0], self.memory[idx][1]

class ActionAwareDictionary:

    def __init__(self):
        self.aware_id2act = ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']
        self.aware_act2id = {v: k for k, v in enumerate(self.aware_id2act)}

    def encode(self, msg):
        if msg in self.aware_act2id:
            return self.aware_act2id[msg]+1
        return -1

    def decode(self, id):
        return self.aware_id2act[id-1]

    def __len__(self):
        return len(self.aware_id2act) + 1

def collate_fn(batch):
    """ pads the sequences to form tensors
    """
    batch.sort(key=lambda x: -len(x[0]))
    context_lengths = [len(x[0]) for x in batch]
    target_lengths = [len(x[1]) for x in batch]

    max_len_context = len(batch[0][0])
    max_len_target = max([len(x[1]) for x in batch])

    context = [np.array(x[0]) for x in batch]
    target = [np.array(x[1]) for x in batch]
    dialog_idxs = [np.array(x[2]) for x in batch]
    context_words = [x[3] for x in batch]
    target_words = [x[4] for x in batch]

    out_context = np.zeros((len(batch), max_len_context, 3), dtype=int)
    out_target = np.zeros((len(batch), max_len_target), dtype=np.int64)
    out_dialog_idxs = np.zeros((len(batch)), dtype=np.int64)

    for i, x in enumerate(batch):
        out_context[i, 0:len(batch[i][0]), :] = context[i]
        out_target[i, 0:len(batch[i][1])] = target[i]
        out_dialog_idxs[i] = dialog_idxs[i]

    return torch.from_numpy(out_context), torch.from_numpy(out_target), context_lengths, target_lengths, out_dialog_idxs, context_words, target_words

# Filters (out channels, in_channels)
class TalkTheWalk(Dataset):

    def __init__(self, data_dir, set):
        self.dialogues = json.load(open(os.path.join(data_dir, 'talkthewalk.{}.json'.format(set))))
        self.act_aware_dict = ActionAwareDictionary()


        # self.data = list()
        self.memory = list()
        self.w2i = defaultdict(lambda: len(self.w2i))

    def read_dataset(self):


        PAD = self.w2i["<pad>"]  # 0
        UNK = self.w2i["<unk>"]  # 1
        EOS = self.w2i["<eos>"]  # 2
        SOS = self.w2i["<sos>"]  # 3
        dialog_idx = 0
        _ = self.w2i['$t'] # tourist token
        _ = self.w2i['$g'] # guide token

        for config in self.dialogues:
            dialogue_context = [[PAD, PAD, PAD]]
            time = 1
            dialogue_context_prev = [[PAD, PAD, PAD]]
            dialog_idx += 1

            for msg in config['dialog']:
                act = msg['text']
                if act == 'EVALUATE_LOCATION':
                    continue
                act_split = act.split(' ')
                if msg['id'] == 'Tourist':
                    act_id = self.act_aware_dict.encode(act)
                    if act_id < 0:
                        # if len(msg['text'].split(' ')) > min_sent_len:
                        dialogue_context.extend([word, '$t', 't'+str(time)] for word in act_split)
                        _ = self.w2i['t' + str(time)]
                        for w in act_split:
                            _ = self.w2i[w]
                            time += 1
                    else:
                        time += 1
                        continue

                else:
                    dialogue_context.extend([[word, '$g', 't' + str(time)] for word in act_split])
                    for w in act_split:
                        _ = self.w2i[w]
                    _ = self.w2i['t' + str(time)]
                    time += 1

                self.memory.append([dialogue_context_prev, act, dialog_idx])

                dialogue_context_prev = dialogue_context.copy()
                dialogue_context_prev.extend([['$$$$'] * 3])




        return self.memory, self.w2i


