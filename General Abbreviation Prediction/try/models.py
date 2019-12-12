#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
# from crf import CRF
import torch.nn.functional as F
import math


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        if layer_num == 1:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)

        else:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, dropout=dropout_rate,
                                  bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for name, p in self.bilstm._parameters.items():
            if p.dim() > 1:
                bias = math.sqrt(6 / (p.size(0) / 4 + p.size(1)))
                nn.init.uniform_(p, -bias, bias)
            else:
                p.data.zero_()
                # This is the range of indices for our forget gates for each LSTM cell
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, lens):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        '''
        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]
        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens, batch_first=True)
        packed_output, _ = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        recover_index = index.argsort()
        output = output[recover_index]
        return output


class SoftmaxDecoder(nn.Module):
    def __init__(self, label_size, input_dim):
        super(SoftmaxDecoder, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        self.linear = torch.nn.Linear(input_dim, label_size)
        self.init_weights()

    def init_weights(self):
        bias = math.sqrt(6 / (self.linear.weight.size(0) + self.linear.weight.size(1)))
        nn.init.uniform_(self.linear.weight, -bias, bias)

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.linear(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, lens, label_ids=None):
        logits = self.forward_model(inputs)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        predict_mask = (torch.arange(inputs.size(1)).expand(len(lens), inputs.size(1))).to(lens.device) < lens.unsqueeze(1)
        if label_ids is not None:
            # cross entropy loss
            p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
            one_hot_labels = torch.eye(self.label_size)[label_ids].type_as(p)
            losses = -torch.log(torch.sum(one_hot_labels * p, -1))  # (batch_size, max_seq_len)
            masked_losses = torch.masked_select(losses, predict_mask)  # (batch_sum_real_len)
            return masked_losses.sum()
        else:
            return torch.argmax(logits, -1), p

class GAP_Model(nn.Module):
    def __init__(self,char_embed,pos_embed,
                 num_labels,hidden_size,dropout_rate=(0.33, 0.5, (0.33, 0.5)),
                 lstm_layer_num = 1,kernel_step = 3,pos_out_size=100, use_pos= True,
                 freeze=False, use_crf=False):
        super(GAP_Model, self).__init__()
        self.char_embed = nn.Embedding.from_pretrained(char_embed, freeze)
        self.char_embed_size = char_embed.size(-1)
        self.use_pos = use_pos
        if use_pos:
            self.pos_embed = nn.Embedding.from_pretrained(pos_embed, freeze)
            self.pos_embed_size = pos_embed.size(-1)
            # self.poscnn = PosCNN(pos_out_size, (kernel_step, self.pos_embed_size), (2, 0))
            self.bilstm = BiLSTM(self.pos_embed_size + self.char_embed_size, hidden_size, dropout_rate[2][1], lstm_layer_num)
        else:
            self.bilstm = BiLSTM(self.char_embed_size, hidden_size, dropout_rate[2][1], lstm_layer_num)

        self.embed_dropout = nn.Dropout(dropout_rate[0])
        self.out_dropout = nn.Dropout(dropout_rate[1])
        self.rnn_in_dropout = nn.Dropout(dropout_rate[2][0])

        if use_crf:
            pass
            # self.decoder = CRFDecoder(num_labels, hidden_size)
        else:
            self.decoder = SoftmaxDecoder(num_labels, hidden_size)



    def forward(self, char_ids, pos_ids, lens, label_ids=None):
        '''

        :param word_ids: (batch_size, max_seq_len)
        :param pos_ids: (batch_size, max_seq_len)
        :param predict_mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len, max_word_len)
        :return: if labels is None, return preds(batch_size, max_seq_len) and p(batch_size, max_seq_len, num_labels);
                 else return loss (scalar).
        '''
        char_embed = self.char_embed(char_ids)
        if self.pos_embed:
            # reshape pos_embed and apply to CNN
            # pos_embed = self.pos_embed(pos_ids).reshape(-1, pos_ids.size(-1), self.pos_embed_size).unsqueeze(1)
            # pos_embed = self.embed_dropout(
            #     pos_embed)  # a dropout layer applied before character embeddings are input to CNN.
            # pos_embed = self.poscnn(pos_embed)
            # pos_embed = self.pos_embed.reshape(pos_ids.size(0), pos_ids.size(1), -1)
            pos_embed = self.pos_embed
            embed = torch.cat([char_embed, pos_embed], -1)
        else:
            embed = char_embed
        x = self.rnn_in_dropout(embed)
        hidden = self.bilstm(x, lens)  # (batch_size, max_seq_len, hidden_size)
        hidden = self.out_dropout(hidden)
        return self.decoder(hidden, lens, label_ids)