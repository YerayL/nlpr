#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from crf import CRF
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import time

class CopyEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128, bidirectional=True):
        super(CopyEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        print(x)
        # input: [b x seq]
        embedded = self.embed(x)
        print(embedded)
        out, h = self.gru(embedded) # out: [b x seq x hid*2] (biRNN)
        return out, h

class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_oovs=12):
        super(CopyDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time = time.time()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size+hidden_size*2,
            hidden_size=hidden_size, batch_first=True)
        self.max_oovs = max_oovs # largest number of OOVs available per sample

        # weights
        self.Ws = nn.Linear(hidden_size*2, hidden_size) # only used at initial stage
        self.Wo = nn.Linear(hidden_size, vocab_size) # generate mode
        self.Wc = nn.Linear(hidden_size*2, hidden_size) # copy mode
        self.nonlinear = nn.Tanh()

    def forward(self, input_idx, encoded, encoded_idx, prev_state, weighted, order):
        # input_idx(y_(t-1)): [b]			<- idx of next input to the decoder (Variable)
        # encoded: [b x seq x hidden*2]		<- hidden states created at encoder (Variable)
        # encoded_idx: [b x seq]			<- idx of inputs used at encoder (numpy)
        # prev_state(s_(t-1)): [1 x b x hidden]		<- hidden states to be used at decoder (Variable)
        # weighted: [b x 1 x hidden*2]		<- weighted attention of previous state, init with all zeros (Variable)

        # hyperparameters
        start = time.time()
        time_check = False
        b = encoded.size(0) # batch size
        seq = encoded.size(1) # input sequence length
        vocab_size = self.vocab_size
        hidden_size = self.hidden_size

        # 0. set initial state s0 and initial attention (blank)
        if order==0:
            prev_state = self.Ws(encoded[:,-1])
            weighted = torch.Tensor(b,1,hidden_size*2).zero_()
            weighted = self.to_cuda(weighted)
            weighted = Variable(weighted)

        prev_state = prev_state.unsqueeze(0) # [1 x b x hidden]
        if time_check:
            self.elapsed_time('state 0')

        # 1. update states
        gru_input = torch.cat([self.embed(input_idx).unsqueeze(1), weighted],2) # [b x 1 x (h*2+emb)]
        _, state = self.gru(gru_input, prev_state)
        state = state.squeeze() # [b x h]

        if time_check:
            self.elapsed_time('state 1')

        # 2. predict next word y_t
        # 2-1) get scores score_g for generation- mode
        score_g = self.Wo(state) # [b x vocab_size]

        if time_check:
            self.elapsed_time('state 2-1')

        # 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
        score_c = F.tanh(self.Wc(encoded.contiguous().view(-1,hidden_size*2))) # [b*seq x hidden_size]
        score_c = score_c.view(b,-1,hidden_size) # [b x seq x hidden_size]
        score_c = torch.bmm(score_c, state.unsqueeze(2)).squeeze() # [b x seq]

        score_c = F.tanh(score_c) # purely optional....

        encoded_mask = torch.Tensor(np.array(encoded_idx==0, dtype=float)*(-1000)) # [b x seq]
        encoded_mask = self.to_cuda(encoded_mask)
        encoded_mask = Variable(encoded_mask)
        score_c = score_c + encoded_mask # padded parts will get close to 0 when applying softmax

        if time_check:
            self.elapsed_time('state 2-2')

        # 2-3) get softmax-ed probabilities
        score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
        probs = F.softmax(score)
        prob_g = probs[:,:vocab_size] # [b x vocab]
        prob_c = probs[:,vocab_size:] # [b x seq]

        if time_check:
            self.elapsed_time('state 2-3')

        # 2-4) add empty sizes to prob_g which correspond to the probability of obtaining OOV words
        oovs = Variable(torch.Tensor(b,self.max_oovs).zero_())+1e-4
        oovs = self.to_cuda(oovs)
        prob_g = torch.cat([prob_g,oovs],1)

        if time_check:
            self.elapsed_time('state 2-4')

        # 2-5) add prob_c to prob_g
        # prob_c_to_g = self.to_cuda(torch.Tensor(prob_g.size()).zero_())
        # prob_c_to_g = Variable(prob_c_to_g)
        # for b_idx in range(b): # for each sequence in batch
        # 	for s_idx in range(seq):
        # 		prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]=prob_c_to_g[b_idx,encoded_idx[b_idx,s_idx]]+prob_c[b_idx,s_idx]


        # prob_c_to_g = Variable
        en = torch.LongTensor(encoded_idx) # [b x in_seq]
        en.unsqueeze_(2) # [b x in_seq x 1]
        one_hot = torch.FloatTensor(en.size(0),en.size(1),prob_g.size(1)).zero_() # [b x in_seq x vocab]
        one_hot.scatter_(2,en,1) # one hot tensor: [b x seq x vocab]
        one_hot = self.to_cuda(one_hot)
        prob_c_to_g = torch.bmm(prob_c.unsqueeze(1),Variable(one_hot, requires_grad=False)) # [b x 1 x vocab]
        prob_c_to_g = prob_c_to_g.squeeze() # [b x vocab]

        out = prob_g + prob_c_to_g
        out = out.unsqueeze(1) # [b x 1 x vocab]

        if time_check:
            self.elapsed_time('state 2-5')

        # 3. get weighted attention to use for predicting next word
        # 3-1) get tensor that shows whether each decoder input has previously appeared in the encoder
        idx_from_input = []
        for i,j in enumerate(encoded_idx):
            idx_from_input.append([int(k==input_idx[i].data[0]) for k in j])
        idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
        # idx_from_input : np.array of [b x seq]
        idx_from_input = self.to_cuda(idx_from_input)
        idx_from_input = Variable(idx_from_input)
        for i in range(b):
            if idx_from_input[i].sum().data[0]>1:
                idx_from_input[i] = idx_from_input[i]/idx_from_input[i].sum().data[0]

        if time_check:
            self.elapsed_time('state 3-1')

        # 3-2) multiply with prob_c to get final weighted representation
        attn = prob_c * idx_from_input
        # for i in range(b):
        # 	tmp_sum = attn[i].sum()
        # 	if (tmp_sum.data[0]>1e-6):
        # 		attn[i] = attn[i] / tmp_sum.data[0]
        attn = attn.unsqueeze(1) # [b x 1 x seq]
        weighted = torch.bmm(attn, encoded) # weighted: [b x 1 x hidden*2]

        if time_check:
            self.elapsed_time('state 3-2')

        return out, state, weighted

    def to_cuda(self, tensor):
        # turns to cuda
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor

    def elapsed_time(self, state):
        elapsed = time.time()
        print("Time difference from %s: %1.4f"%(state,elapsed-self.time))
        self.time = elapsed
        return

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
        """
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        """
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

    # lens
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

class CRFDecoder(nn.Module):
    def __init__(self, label_size, input_dim):
        super(CRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=label_size)
        self.crf = CRF(label_size + 2)
        self.label_size = label_size

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

    def forward(self, inputs, lens, labels=None):
        '''
        :param inputs:(batch_size, max_seq_len, input_dim)
        :param predict_mask:(batch_size, max_seq_len)
        :param labels:(batch_size, max_seq_len)
        :return: if labels is None, return preds(batch_size, max_seq_len) and p(batch_size, max_seq_len, num_labels);
                 else return loss (scalar).
        '''
        logits = self.forward_model(inputs)  # (batch_size, max_seq_len, num_labels)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        logits = self.crf.pad_logits(logits)
        predict_mask = (torch.arange(inputs.size(1)).expand(len(lens), inputs.size(1))).to(lens.device) < lens.unsqueeze(1)
        if labels is None:
            _, preds = self.crf.viterbi_decode(logits, predict_mask)
            return preds, p
        return self.neg_log_likehood(logits, predict_mask, labels)

    def neg_log_likehood(self, logits, predict_mask, labels):
        norm_score = self.crf.calc_norm_score(logits, predict_mask)
        gold_score = self.crf.calc_gold_score(logits, labels, predict_mask)
        loglik = gold_score - norm_score
        return -loglik.sum()

class GAP_Model(nn.Module):
    def __init__(self, char_embed, pos_embed,
                 num_labels, hidden_size, dropout_rate=(0.33, 0.5, (0.33, 0.5)),
                 lstm_layer_num=1, kernel_step = 3, pos_out_size=100, use_pos=True,
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
            self.decoder = CRFDecoder(num_labels, hidden_size)
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
            pos_embed = self.pos_embed(pos_ids)
            embed = torch.cat([char_embed, pos_embed], -1)
        else:
            embed = char_embed
        x = self.rnn_in_dropout(embed)
        hidden = self.bilstm(x, lens)  # (batch_size, max_seq_len, hidden_size)
        hidden = self.out_dropout(hidden)
        return self.decoder(hidden, lens, label_ids)