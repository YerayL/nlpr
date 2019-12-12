#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.optim as optim
from tqdm import tqdm
from torchtext.vocab import Vectors
from models import GAP_Model
import codecs
from util import load_iters, get_chunks


torch.manual_seed(1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "data"
PREDICT_OUT_FILE = "res2.txt"
BEST_MODEL = "best_model2.ckpt"
BATCH_SIZE = 10
EPOCHS = 200

# embedding
CHAR_VECTORS = None
# CHAR_VECTORS = Vectors('glove.6B.100d.txt', '../../embeddings/glove.6B')
CHAR_EMBEDDING_SIZE = 50 # 100
POS_VECTORS = None
POS_EMBEDDING_SIZE = 40 # 30  # the input char embedding to CNN
FREEZE_EMBEDDING = False

# SGD parameters
LEARNING_RATE = 0.015
DECAY_RATE = 0.05
MOMENTUM = 0.9
CLIP = 5
PATIENCE = 5

# network parameters
HIDDEN_SIZE = 400  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
LSTM_LAYER_NUM = 1
DROPOUT_RATE = (0.5, 0.5, (0.5, 0.5))  # after embed layer, other case, (input to rnn, between rnn layers)
USE_POS = True  # use char level information
# N_FILTERS = 30  # the output char embedding from CNN
# KERNEL_STEP = 3  # n-gram size of CNN
USE_CRF = True

def train(train_iter, dev_iter, optimizer):
    best_dev_f1 = [0.93, 0.56, 0.92]
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        train_iter.init_epoch()
        for i, batch in enumerate(tqdm(train_iter)):
            chars, lens = batch.char
            labels = batch.label
            if i < 2:
                tqdm.write(' '.join([CHAR.vocab.itos[i] for i in chars[0]]))
                tqdm.write(' '.join([LABEL.vocab.itos[i] for i in labels[0]]))
            model.zero_grad()
            loss = model(chars, batch.pos, lens, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch, total_loss))

        lr = LEARNING_RATE / (1 + DECAY_RATE * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        dev_f1 = eval(dev_iter, "Dev", epoch)
        if dev_f1[0] < best_dev_f1[0] and dev_f1[1] < best_dev_f1[1] and dev_f1[2] < best_dev_f1[2]:
            patience_counter += 1
            tqdm.write("No improvement, patience: %d/%d" % (patience_counter, PATIENCE))
        else:
            for i,k in enumerate(dev_f1):
                if k > best_dev_f1[i]:
                    best_dev_f1[i] = k
            # best_dev_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL)
            tqdm.write("New best model, saved to best_model.ckpt, patience: 0/%d " % PATIENCE)
            print('best d_acc, all_acc, char_acc', best_dev_f1 )
        if patience_counter >= PATIENCE:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break



def eval(data_iter, name, epoch=None, best_model=None):
    if best_model:
        model.load_state_dict(torch.load(best_model))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct_out = 0
        total_full = 0
        correct_label = 0
        total_cha = 0
        discriminate = 0
        for i, batch in enumerate(data_iter):
            chars, lens = batch.char
            labels = batch.label
            predicted_seq, _ = model(chars, batch.pos, lens)
            loss = model(chars, batch.pos, lens, labels)
            total_loss += loss.item()

            orig_text = [e.char for e in data_iter.dataset.examples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
            for text, ground_truth_id, predict_id, len_ in zip(orig_text, labels.cpu().numpy(),
                                                               predicted_seq.cpu().numpy(),
                                                               lens.cpu().numpy()):
                total_cha += len_
                total_full += 1
                if (ground_truth_id[:len_] == predict_id[:len_]).all():
                    correct_out += 1
                    discriminate += 1
                    correct_label += len_
                else:
                    for i in predict_id[:len_]:
                        if i in ground_truth_id[:len_]:
                            correct_label += 1
                    if not (list(set(ground_truth_id[:len_])) == [3] and list(set(ground_truth_id[:len_])) != list(set(predict_id[:len_]))):
                        discriminate += 1

        # Calculating the F1-Score
        d_acc = discriminate / total_full if total_full != 0 else 0
        all_acc = correct_out / total_full if total_full != 0 else 0
        char_acc = correct_label / total_cha if total_cha != 0 else 0

        if epoch is not None:
            tqdm.write(
                "Epoch: %d, %s  d_acc : %.3f,all_acc : %.3f,char_acc : %.3f, Loss %.3f" % (epoch, name, d_acc, all_acc,char_acc,
                                                                                         total_loss))
        else:
            tqdm.write(
                " %s  d_acc : %.3f,all_acc : %.3f,char_acc : %.3f, Loss %.3f" % (name, d_acc, all_acc, char_acc,
                                                                               total_loss))
    return d_acc, all_acc, char_acc




def predict(data_iter, out_file):
    model.eval()
    with torch.no_grad():
        gold_seqs = []
        predicted_seqs = []
        char_seqs = []
        for i, batch in enumerate(data_iter):
            chars, lens = batch.char
            predicted_seq, _ = model(chars, batch.char, lens)
            gold_seqs.extend(batch.label.tolist())
            predicted_seqs.extend(predicted_seq.tolist())
            char_seqs.extend(chars.tolist())
            write_predicted_labels(out_file, data_iter.dataset.examples, word_seqs, LABEL.vocab.itos, gold_seqs,
                               predicted_seqs)


def write_predicted_labels(output_file, orig_text, word_ids, id2label, gold_seq, predicted_seq):
    with codecs.open(output_file, 'w', encoding='utf-8') as writer:
        for text, wids, predict, gold in zip(orig_text, word_ids, predicted_seq, gold_seq):
            ix = 0
            for w_id, p_id, g_id in zip(wids, predict, gold):
                if w_id == pad_idx: break
                output_line = ' '.join([text.word[ix], id2label[g_id], id2label[p_id]])
                writer.write(output_line + '\n')
                ix += 1
            writer.write('\n')


if __name__ == '__main__':
    train_iter, dev_iter, test_iter, CHAR, POS, LABEL = load_iters(CHAR_EMBEDDING_SIZE, CHAR_VECTORS,
                                                                    POS_EMBEDDING_SIZE, POS_VECTORS,
                                                                    BATCH_SIZE, DEVICE, DATA_PATH)

    model = GAP_Model(CHAR.vocab.vectors,pos_embed=POS.vocab.vectors,num_labels=len(LABEL.vocab.stoi),hidden_size=HIDDEN_SIZE,dropout_rate=DROPOUT_RATE,
                      lstm_layer_num=LSTM_LAYER_NUM,
                      use_pos=USE_POS,freeze=FREEZE_EMBEDDING, use_crf=USE_CRF).to(DEVICE)
    # print(model)
    pad_idx = CHAR.vocab.stoi[CHAR.pad_token]

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # train(train_iter, dev_iter, optimizer)
    eval(test_iter, "Test", best_model=BEST_MODEL)
    # predict(test_iter, PREDICT_OUT_FILE)
