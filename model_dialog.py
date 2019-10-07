

from torch import nn

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler



TYPE = torch.LongTensor
TYPEF = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    TYPE = torch.cuda.LongTensor
    TYPEF = torch.cuda.FloatTensor


################################################

class Model(nn.Module):
    # def __init__(self, gru_size, nwords, lr, hops, dropout, unk_mask):
    def __init__(self, hops, nwords, emb_size, gru_size, w2i):

        super(Model, self).__init__()
        self.name = "Mem2Seq"
        self.nwords = nwords
        self.gru_size = gru_size
        self.emb_size = emb_size
        assert (self.gru_size == self.emb_size)

        self.hops = hops
        self.w2i = w2i
        self.i2w = {v: k for k, v in self.w2i.items()}

        self.encoder = Encoder(hops, nwords, gru_size)
        self.decoder = Decoder(emb_size, hops, gru_size, nwords)

        self.optim_enc = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=0.001)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim_dec, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()  # masked_cross_entropy

        if use_cuda:
            self.cross_entropy = self.cross_entropy.cuda()
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.loss = 0
        self.ploss = 0
        self.vloss = 0
        self.n = 1
        self.batch_size = 0
        self.dropout = 0.2

    def print_loss(self):
        print_loss_avg = self.loss / self.n
        print_acc_avg = self.acc / self.n
        self.n += 1
        return 'L:{:.5f}, A:{:.5f}'.format(print_loss_avg, print_acc_avg)

    def losses(self):
        return self.loss / self.n

    def accs(self):
        return self.acc / self.n

    def train_batch(self, context, responses, new_epoch, context_lengths, target_lengths, clip_grads):

        # (TODO): remove transpose
        if new_epoch:  # (TODO): Change this part
            self.loss = 0
            self.acc = 0
            self.n = 1


        context = context.type(TYPE)
        responses = responses.type(TYPE)

        self.optim_enc.zero_grad()
        self.optim_dec.zero_grad()

        h = self.encoder(context.transpose(0, 1))
        self.decoder.load_memory(context.transpose(0, 1))
        y = torch.from_numpy(np.array([2] * context.size(1), dtype=int)).type(TYPE)
        y_len = 0

        h = h.unsqueeze(0)
        output_vocab = torch.zeros(max(target_lengths), context.size(1), self.nwords)
        while y_len < responses.size(0):  # TODO: Add EOS condition
            p_vocab, h = self.decoder(context, y, h)
            output_vocab[y_len] = p_vocab

            y = responses[y_len].type(TYPE)
            y_len += 1

        mask_v = torch.ones(output_vocab.size())

        for i in range(responses.size(1)):
            mask_v[target_lengths[i]:, i, :] = 0

        loss_v = self.cross_entropy(output_vocab.contiguous().view(-1, self.nwords),
                                    responses.cpu().contiguous().view(-1))

        loss = loss_v

        loss.backward()
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 10.0)
        self.optim_enc.step()
        self.optim_dec.step()

        self.loss += loss.item()

        return loss.item()



    def evaluate_batch(self, batch_size, input_batches, target_lengths, src_plain):

        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches.transpose(0, 1)).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([2] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(max(target_lengths), batch_size, self.nwords))

        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)

        self.from_whichs = []
        for t in range(max(target_lengths)):
            decoder_vocab, decoder_hidden = self.decoder(input_batches, decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vocab
            topv, topvi = decoder_vocab.data.topk(1)
            next_in = [topvi[i].item() for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input

            temp = []
            for i in range(batch_size):
                    ind = topvi[i].item()
                    if ind == 3:
                        temp.append('<eos>')
                    else:
                        temp.append(self.i2w[ind])
            decoded_words.append(temp)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words

class Encoder(nn.Module):
    def __init__(self, hops, nwords, emb_size):
        super(Encoder, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data = torch.normal(0.0, torch.ones(self.nwords, self.emb_size) * 0.1)

        self.hops = hops
        self.nwords = nwords
        self.emb_size = emb_size
        self.dropout = 0.2

        # (TODO) : Initialize with word2vec
        self.A = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.A.apply(init_weights)
        self.C = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.C.apply(init_weights)
        for i in range(self.hops - 1):
            self.C[i].weight = self.A[i + 1].weight
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, context):
        # (TODO): Use pack_padded_sequence
        size = context.size()  # b x l x 3

        if (self.training):  ### Dropout
            ones = np.ones((size[0], size[1], size[2]))
            rand_mask = np.random.binomial([np.ones((size[0], size[1]))], 1 - self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            a = Variable(torch.Tensor(ones))
            if use_cuda: a = a.cuda()
            context = context * a.long()

        q = torch.zeros(size[0], self.emb_size).type(TYPEF)  # initialize u # batchsize x length x emb_size
        q_list = [q]

        context = context.contiguous().view(size[0], -1)  # b x l*3
        for h in range(self.hops):
            m = self.A[h](context)  # b x l*3 x e
            m = m.view(size[0], size[1], size[2], self.emb_size)  # b x l x 3 x e
            m = torch.sum(m, 2)  # b x l x e
            p = torch.sum(m * q.unsqueeze(1), 2)  # b x l (TODO): expand_as(m)
            attn = self.soft(p)

            c = self.C[h](context)  # b x l*3 x e
            c = c.view(size[0], size[1], size[2], self.emb_size)  # b x l x 3 x e
            c = torch.sum(c, 2).squeeze(2)  # b x l x e
            o = torch.bmm(attn.unsqueeze(1), c).squeeze(1)
            q = q + o
        return q


class Decoder(nn.Module):
    # def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
    def __init__(self, emb_size, hops, gru_size, nwords):

        super(Decoder, self).__init__()
        self.nwords = nwords
        self.hops = hops
        self.emb_size = emb_size
        self.gru_size = gru_size
        self.dropout = 0.2

        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data = torch.normal(0.0, torch.ones(self.nwords, self.emb_size) * 0.1)

        self.A = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.A.apply(init_weights)
        self.C = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.C.apply(init_weights)
        for i in range(self.hops - 1):
            self.C[i].weight = self.A[i + 1].weight

        self.soft = nn.Softmax(dim=1)
        self.lin_vocab = nn.Linear(2 * emb_size, self.nwords)
        self.gru = nn.GRU(emb_size, emb_size, dropout=self.dropout)

    def load_memory(self, context):
        size = context.size()  # b * m * 3

        if (self.training):
            ones = np.ones((size[0], size[1], size[2]))
            rand_mask = np.random.binomial([np.ones((size[0], size[1]))], 1 - self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            a = Variable(torch.Tensor(ones))
            if use_cuda:
                a = a.cuda()
            context = context * a.long()

        self.memories = []
        context = context.view(size[0], -1)
        for hop in range(self.hops):
            m = self.A[hop](context)
            m = m.view(size[0], size[1], size[2], self.emb_size)
            m = torch.sum(m, 2)
            self.memories.append(m)
            c = self.C[hop](context)
            c = c.view(size[0], size[1], size[2], self.emb_size)
            c = torch.sum(c, 2)
        self.memories.append(c)

    def forward(self, context, y_, h_):
        m = self.C[0](y_).unsqueeze(0)  # b * e
        _, h = self.gru(m, h_)

        q = [h.squeeze(0)]
        for hop in range(self.hops):
            p = torch.sum(self.memories[hop] * q[-1].unsqueeze(1).expand_as(self.memories[hop]), 2)
            attn = self.soft(p)
            o = torch.bmm(attn.unsqueeze(1), self.memories[hop + 1]).squeeze(1)
            q.append(q[-1] + o)
            if hop == 0:
                p_vocab = self.lin_vocab(torch.cat((q[0], o), 1))

        p_ptr = p
        return p_vocab, h