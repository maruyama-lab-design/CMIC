import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy import data
from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator, BucketIterator

from torchtext.vocab import Vectors

import pandas as pd
import time
import math
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse

torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

'''
Train and validate the MethGRU model.

INPUT:  random-length k-mer sequences and corresponding label[M2M(1) or M2U(0)]  (trv1data.csv & trv2data.csv & trv3data.csv & trv4data.csv & trv5data.csv)
OUTPUT: train and validation result (filetitle.csv)
        trained MethGRU model (H_epoch_dp_lr_filetitle_model.pt)

'''

def pack_padded_sequence_202109(input, text_lengths, batch_first=False, enforce_sorted=True):
    text_lengths = torch.as_tensor(text_lengths, dtype=torch.int64) # The data, text_lengths, is converted into a torch.Tensor. 
    text_lengths = text_lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        text_lengths, sorted_indices = torch.sort(text_lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(input, text_lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)



class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # fully connect layer
        self.act = nn.Sigmoid() # activate function

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        # packed_embedded = nn.utils.rnn.\
        #     pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_embedded = \
            pack_padded_sequence_202109(embedded, text_lengths, batch_first=True)

        packed_output, hidden = self.gru(packed_embedded)

        # Concatenation of the final forward and backward hidden states. 
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 

        dense_outputs = self.fc(hidden) # Fully connect. 
        outputs = self.act(dense_outputs)

        # out_prob = F.softmax(outputs.view(batch_size, -1))

        return outputs


def count_params(model):
    return sum(param.numel() for param in model.parameters() \
        if param.requires_grad)

def calc_precision(TP, FP):
    precision = 0
    if TP + FP > 0:
        precision = TP / (TP + FP)
    return precision
def calc_recall(TP, FN):
    recall = 0
    if TP + FN > 0:
        recall = TP / (TP + FN)
    return recall
def calc_FPR(TP, TN, FP):
    FPR = 0
    if TP + TN > 0:
        FPR = FP / (TP + TN)
    return FPR
def calc_F(precision, recall):
    F = 0
    if precision + recall > 0:
        F = (2 * precision * recall) / (precision + recall)
    return F
    
def performance(preds, y):
    # round to the nearest whole number 0 or 1
    # the prediction result of a batch
    TP = FP = FN = TN = 0
    rounded_preds = torch.round(preds)
    for i in range(len(rounded_preds)):
        if rounded_preds[i] == 1 and y[i] == 1:
            TP = TP + 1
        elif rounded_preds[i] == 1 and y[i] == 0:
            FP = FP + 1
        elif rounded_preds[i] == 0 and y[i] == 1:
            FN = FN + 1
        else:
            TN = TN + 1
    acc = (TN + TP) / (TP + TN + FP + FN)
    precision = calc_precision(TP, FP)
    recall = calc_recall(TP, FN)
    F = calc_F(precision, recall)
    FPR = calc_FPR(TP, TN, FP)
    return acc, precision, recall, F, FPR


def test_performance(pred_y, true_y):
    # round to the nearest whole number 0 or 1
    # the prediction result of a batch
    TP = FP = FN = TN = 0
    rounded_preds = torch.round(pred_y)
    for i in range(len(rounded_preds)):
        if rounded_preds[i] == 1 and true_y[i] == 1:
            TP = TP + 1
        elif rounded_preds[i] == 1 and true_y[i] == 0:
            FP = FP + 1
        elif rounded_preds[i] == 0 and true_y[i] == 1:
            FN = FN + 1
        else:
            TN = TN + 1
    acc = (TN + TP) / (TP + TN + FP + FN)

    precision = calc_precision(TP, FP)
    recall = calc_recall(TP, FN)
    F = calc_F(precision, recall)
    FPR = calc_FPR(TP, TN, FP)

    tensor_preds = pred_y.cpu()
    tensor_y = true_y.cpu()
    preds = tensor_preds.detach().numpy().tolist()
    y = tensor_y.detach().numpy().tolist()
    return acc, precision, recall, F, FPR, preds, y


#
# def ROC(pred_y, true_y):
#     fpr, tpr, thresholds = roc_curve(true_y,pred_y)
#     plt.plot(fpr, tpr, marker='o')
#     plt.xlabel('FPR: False positive rate')
#     plt.ylabel('TPR: True positive rate')
#     plt.title('')
#     plt.grid()
#     plt.savefig('sklearn_roc_curve.png')
#
#     fpr_all, tpr_all, thresholds_all = roc_curve(true_y,pred_y,
#                                                  drop_intermediate=False)
#     plt.plot(fpr_all, tpr_all, marker='o')
#     plt.xlabel('FPR: False positive rate')
#     plt.ylabel('TPR: True positive rate')
#     plt.grid()
#     plt.savefig('sklearn_roc_curve_all.png')

def plot_roc(predict_prob, labels, kind):
    # false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    # roc_auc=auc(false_positive_rate, true_positive_rate)
    # plt.title(kind+'ROC curve of test result')
    # plt.plot(false_positive_rate, true_positive_rate,label='AUC = %0.4f'% roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0,1],[0,1],'r--')
    # plt.ylabel('TPR: True positive rate')
    # plt.xlabel('FPR: False positive rate')
    # plt.grid(linestyle = '--',alpha = 0.3)
    # plt.savefig(kind+'sklearn_roc_curve.png')

    fpr_all, tpr_all, thresholds_all = roc_curve(labels, predict_prob,
                                                 drop_intermediate=False)
    roc_auc = auc(fpr_all, tpr_all)
    plt.title('ROC curve of test result')
    plt.plot(fpr_all, tpr_all, label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR: True positive rate')
    plt.xlabel('FPR: False positive rate')
    plt.grid(linestyle='--', alpha=0.3)
    plt.savefig(kind + 'all_sklearn_roc_curve.png')


def train(model, iterator, optimizer, criterion):
    # initalize every epoch
    epoch_loss = 0
    epoch_acc = epoch_pred = epoch_recall = epoch_F = epoch_FPR = 0

    # set the model in training phase
    model.train()

    for batch in iterator:
        optimizer.zero_grad() # resets the gradients
        text, text_lengths = batch.CGI_seq # get text and the number of words
        predictions = model(text, text_lengths).squeeze()         # convert to 1D tensor
        loss = criterion(predictions, batch.Label) 
        acc, pred, recall, F, FPR = performance(predictions, batch.Label)
        loss.backward()  # The loss is backpropaged, and the gradients are computed. 
        optimizer.step() # the weights are updated. 

        # loss and performance scores
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_pred += pred
        epoch_recall += recall
        epoch_F += F
        epoch_FPR += FPR

        # print('Train batch')
        # torch.save(model, 'saved_weights.pkl')

    return round((epoch_loss / len(iterator)), 3), round((epoch_acc / len(iterator)), 3), round(
        (epoch_pred / len(iterator)), 3), round((epoch_recall / len(
        iterator)), 3), round((epoch_F / len(iterator)), 3), round((epoch_FPR / len(iterator)), 3)


def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = epoch_pred = epoch_recall = epoch_F = epoch_FPR = 0

    # deactivating dropout layer
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.CGI_seq

            # 转换为一维张量
            predictions = model(text, text_lengths).squeeze()

            # 计算损失和准确性
            loss = criterion(predictions, batch.Label)
            acc, pred, recall, F, FPR = performance(predictions, batch.Label)

            # 跟踪损失和准确性

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_pred += pred
            epoch_recall += recall
            epoch_F += F
            epoch_FPR += FPR

    return round((epoch_loss / len(iterator)), 3), round((epoch_acc / len(iterator)), 3), round(
        (epoch_pred / len(iterator)), 3), round((epoch_recall / len(
        iterator)), 3), round((epoch_F / len(iterator)), 3), round((epoch_FPR / len(iterator)), 3)


def testmodel(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = epoch_pred = epoch_recall = epoch_F = epoch_FPR = 0
    test_pred = []
    test_y = []

    # deactivating dropout layer
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.CGI_seq

            # 转换为一维张量
            predictions = model(text, text_lengths).squeeze()

            # 计算损失和准确性
            loss = criterion(predictions, batch.Label)
            acc, pred, recall, F, FPR, preds, y = test_performance(predictions, batch.Label)

            # 跟踪损失和准确性

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_pred += pred
            epoch_recall += recall
            epoch_F += F
            epoch_FPR += FPR
            test_pred.append(preds)
            test_y.append(y)
    return round((epoch_loss / len(iterator)), 3), round((epoch_acc / len(iterator)), 3), round(
        (epoch_pred / len(iterator)), 3), round((epoch_recall / len(
        iterator)), 3), round((epoch_F / len(iterator)), 3), round((epoch_FPR / len(iterator)), 3), test_pred, test_y


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def standardloss(l):
    arr = np.array(l)
    arr_sl = np.std(arr, ddof=1)
    arr_sme = arr_sl / np.sqrt(len(arr))
    return arr_sme


def average(l):
    arr = np.array(l)
    avg = sum(arr) / len(arr)
    return avg


def final(result, fd, epoch):
    final = []
    for i in range(fd):
        temps = result[i]
        temp = temps[epoch - 1]
        final.append(temp)
    return final


def starttrain(allcgikmercsv, traincsv, vldcsv, modelpath, emd, Epoch, hidden_node, dp, learnratio, csvpath, decay,
               filetitle, kind):
    datafields = [("idx", None),  # we won't be needing the idx, so we pass in None as the field
                  ("CGI_seq", TEXT),
                  ("Label", LABEL)]

    trn = TabularDataset(
        path=traincsv,  # the root directory where the data lies
        format='csv',
        skip_header=True,
        # if csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data
        fields=datafields)

    vld = TabularDataset(
        path=vldcsv,  # the root directory where the data lies
        format='csv',
        skip_header=True,
        # if csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data
        fields=datafields)

    # add new
    allcgi = TabularDataset(
        path=allcgikmercsv,  # the root directory where the data lies
        format='csv',
        skip_header=True,
        # if csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data
        fields=datafields)

    # trn, vld = trnvld.split(split_ratio=0.7, random_state=random.seed(SEED))

    # read in pretrained embedding vector
    nm = 'dna2vec.txt'
    c = csvpath + '/'

    vectors = Vectors(name=nm, cache=c)
    TEXT.build_vocab(allcgi, vectors=vectors)
    LABEL.build_vocab(trn)
    vo = len(TEXT.vocab)
    print("Size of TEXT vocabulary:", len(TEXT.vocab))
    print("Size of LABEL vocabulary:", len(LABEL.vocab))
    recordrank = TEXT.vocab.freqs.most_common(len(TEXT.vocab))
    # print(recordrank)
    file = open(csvpath + '/frequencerank_' + filetitle + '.txt', 'w')
    file.write(str(recordrank))
    file.close()

    print(TEXT.vocab.stoi)
    # print(TEXT.vocab.vectors)
    # print(TEXT.vocab.stoi['GC'])
    # print(TEXT.vocab.vectors[2])

    # ------------Declaring the Iter---------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 32

    train_iter = BucketIterator(
        trn,  # we pass in the datasets we want the iterator to draw data from
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.CGI_seq),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        device=device)

    valid_iter = BucketIterator(
        vld,  # we pass in the datasets we want the iterator to draw data from
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.CGI_seq),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        device=device)

    # hyperparameters of RNN
    size_of_vocab = len(TEXT.vocab)
    embedding_dim = emd
    num_hidden_nodes = hidden_node
    num_output_nodes = 1
    num_layers = 1
    dropout = dp

    # model object
    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                       bidirectional=True, dropout=dropout)

    # using pretrained embedding vectors?
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    # uniform distribution
    # model.embedding.weight.data.uniform_(-1,1)
    # print(model.embedding.weight.shape)
    # print(model.embedding.weight)

    # embedding layer trainable?
    model.embedding.weight.requires_grad = False
    # model.embedding.weight.requires_grad = True

    # if kind == 1:
    #     model.embedding.weight.requires_grad = False
    # elif kind == 2:
    #     model.embedding.weight.requires_grad = True
    # elif kind == 3:
    #     model.embedding.weight.requires_grad = False
    #     model.embedding.weight.data.uniform_(-1, 1)
    # else:
    #     model.embedding.weight.requires_grad = True
    #     model.embedding.weight.data.uniform_(-1, 1)

    # print(pretrained_embeddings.shape)
    # print(model.embedding.weight)
    print(f'The model has {count_params(model):,} trainable parameters')

    # define optimizier
    optimizer = optim.Adam(model.parameters(), lr=learnratio, weight_decay=decay)

    criterion = nn.BCELoss()

    # use GPU
    model = model.to(device)
    criterion = criterion.to(device)
    print('------------ Starting  trainning ---------')

    # trainning parameters
    N_EPOCHS = Epoch
    start = time.time()

    print('------------ Trainning ---------')

    # __________train______________

    all_losses = []
    all_train_acc = []
    all_train_pre = []
    all_train_recall = []
    all_train_F = []
    all_train_FPR = []

    all_vld_losses = []
    all_vld_acc = []
    all_vld_pre = []
    all_vld_recall = []
    all_vld_F = []
    all_vld_FPR = []

    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        print('Strat train epoch =', epoch)
        # 训练模型
        train_loss, train_acc, train_pre, train_recall, train_F, train_FPR = train(model, train_iter, optimizer,
                                                                                   criterion)
        all_losses.append(train_loss)
        all_train_acc.append(train_acc)
        all_train_pre.append(train_pre)
        all_train_recall.append(train_recall)
        all_train_F.append(train_F)

        valid_loss, valid_acc, valid_pre, valid_recall, valid_F, valid_FPR = evaluate(model, valid_iter, criterion)
        all_vld_losses.append(valid_loss)
        all_vld_acc.append(valid_acc)
        all_vld_pre.append(valid_pre)
        all_vld_recall.append(valid_recall)
        all_vld_F.append(valid_F)

        print('Train epoch:', epoch, 'is over.  ', 'Train time = ',
              timeSince(start), 'Loss = ', train_loss, 'Train accuracy = ', train_acc,
              'Train predicitin = ', train_pre,
              'Train recall = ', train_recall, 'F-value = ', train_F)
        print('Validation epoch:', epoch, 'is over.  ', 'Validate time = ',
              timeSince(start), 'Loss = ', valid_loss, 'Validate accuracy = ', valid_acc,
              'Validate predicitin = ', valid_pre,
              'Validate recall = ', valid_recall, 'Validate F-value = ', valid_F)

        # torch.save(model.state_dict(), modelpath)
        # # using early stopping
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), modelpath)
        # else:
        #     print('Early stopping')
        #     break
        # N = baocun cishu
        torch.save(model.state_dict(), modelpath)

    # _______record trained embedding vector_____
    emd_we = model.embedding.weight
    emd_we_cpu = emd_we.cpu()
    emd_we_record = emd_we_cpu.detach().numpy()
    np.savetxt(csvpath + '/' + filetitle + '_emd.txt', emd_we_record)

    # __________test______________
    #
    # print('---------test----------')
    # model.load_state_dict(torch.load(modelpath))
    # test_loss, test_acc, test_pred, test_recall, test_F,test_FPR,test_preds,test_ys = testmodel(model, test_iter, criterion)
    # print('Test loss is:',test_loss)
    # print('Test accuracy is :',test_acc)
    # print('Test precision is:',test_pred)
    # print('Test recall is:',test_recall)
    # print('Test F-value is:',test_F)
    #
    # preds_for_plot = []
    # for i in range(len(test_preds)):
    #     temp = test_preds[i]
    #     for j in range(len(temp)):
    #         preds_for_plot.append(temp[j])
    # ys_for_plot = []
    # for i in range(len(test_ys)):
    #     temp = test_ys[i]
    #     for j in range(len(temp)):
    #         ys_for_plot.append(temp[j])
    # plot_roc(preds_for_plot, ys_for_plot,kind)

    return all_losses, all_train_acc, all_train_pre, all_train_recall, all_train_F, \
           all_vld_losses, all_vld_acc, all_vld_pre, all_vld_recall, all_vld_F, vo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='default values of hyperparameters of CMIC')
    parser.add_argument('--SEED', type=int, help='random seed for train', default=2020)
    parser.add_argument('--N', type=int, help='time of data augmentation', default=1)
    parser.add_argument('--f', type=int, help='N-fold cross validation', default=5)
    parser.add_argument('--w', type=int, help='neighbour window in word2vec', default=10)
    parser.add_argument('--epoch', type=int, help='train and validation epoch', default=10)
    parser.add_argument('--decay', type=float, help='weight decay rate', default=0.01)
    parser.add_argument('--H', type=int, help='hidden state dimension of RNN', default=256)
    parser.add_argument('--dp', type=float, help='dropout rate', default=0.5)
    parser.add_argument('--D', type=int, help='word embedding vector dimension', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=10e-5)
    parser.add_argument('--cell_type', help='cell type', default='FGO2BM')
    parser.add_argument('--rc', help='use reverse complment or not', default=True)
    parser.add_argument('--mode', help='mode of generating kmer sequences', choices=[0, 1], default=1)
    parser.add_argument('--title', help='filename of input', default='FGO2BMCGI')
    # parser.add_argument('--time_generateRLKS', help='filename of input', default='2021-03-16')
    parser.add_argument('--time_generateRLKS', help='filename of input', default='2021-03-20')

    parser.add_argument('--filetitle', help='title of result file', default='5fcv')

    parser.add_argument('--input_dir', help='It looks like FGO2PB/2021-09-06')

    args = parser.parse_args()

    # Please rewrite your rootpath!!!
    # rootpath = '../data/' + args.title + '/' + args.time_generateRLKS + '/'
    # rootpath = os.path.join(args.input_dir) 
    
    # rootpath = "/mnt/c/Users/osamu/OneDrive - Kyushu University/work/methyl-word2vec-RNN/CMIC/data/FGO2PB/2021-09-03"
    # rootpath = "/mnt/c/Users/om/OneDrive - Kyushu University/work/methyl-word2vec-RNN/CMIC/data/FGO2PB/2021-09-03"
    rootpath = args.input_dir
    
    # csvpath: the directory path of input k-mer sequences
    # modelpath: the path of trained model
    # resultcsvpath: the path of result csv file

    torch.manual_seed(args.SEED)

    # Setting for Cuda
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    # ------------Declaring the Fields---------
    tokenize = lambda x: x.split()
    TEXT = Field(tokenize=tokenize, include_lengths=True, batch_first=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    all_kind = ['2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '2_10', '3_5', '3_6', '3_7', '3_8', '3_9', '3_10', '4_6',
                '4_7', '4_8', '4_9', '4_10']
    #all_kind = ['2_4']

    result = pd.DataFrame(
        columns=(
        'type', 'vo', 'embedding_vector_size', 'neighbour_winodw', 'epoch', 'hidden_nodes', 'dropout', 'learning_ratio',
        'weight_decay',
        'train_loss',
        'train_accuary', 'train_precision', 'train_recall', 'train_F-value',
        'validate_loss',
        'validate_accuary', 'validate_precision', 'validate_recall', 'validate_F-value',
        'final_vld_accuary', 'final_vld_precision', 'final_vld_recall', 'final_vld_F-value',
        'average_vld_accuary', 'average_vld_precision', 'average_vld_recall', 'average_vld_F-value',
        'sl_acc', 'sl_prec', 'sl_re', 'sl_F'))
    for c in range(len(all_kind)):
        kind = all_kind[c]

        print('---------------- strat training model for ' + str(kind) + '----------------')

        # csvpath = rootpath + args.cell_type + '_rc_' + str(args.N) + '_' + str(args.mode) + '_' + kind
        csvpath = os.path.join(rootpath, args.cell_type + '_rc_' + str(args.N) + '_' + str(args.mode) + '_' + kind)


        trv1data = csvpath + '/trv1data.csv'
        trv2data = csvpath + '/trv2data.csv'
        trv3data = csvpath + '/trv3data.csv'
        trv4data = csvpath + '/trv4data.csv'
        testcsv = csvpath + '/trv5data.csv'
        allcgikmercsv = csvpath + '/allcgikmer.csv'

        # list for recording
        all_trian_loss = []
        all_train_acc = []
        all_train_pre = []
        all_train_recall = []
        all_train_F = []
        all_vld_loss = []
        all_vld_acc = []
        all_vld_pred = []
        all_vld_recall = []
        all_vld_F = []
        final_vld_acc = []
        final_vld_pred = []
        final_vld_recall = []
        final_vld_F = []
        # ------------Constructing the Dataset------
        modelpath = csvpath + '/result/' + str(args.H) + '_' + str(args.epoch) + '_' + str(args.dp) + '_' + str(
            args.lr) + '_' + args.filetitle + '_model.pt'

        # estdata.to_csv(testcsv, columns=['idx', 'CGI_seq', 'Label'], index=0)
        df1 = pd.read_csv(trv1data)
        df2 = pd.read_csv(trv2data)
        df3 = pd.read_csv(trv3data)
        df4 = pd.read_csv(trv4data)
        df5 = pd.read_csv(testcsv)
        datasets = [df1, df2, df3, df4, df5]
        vldcsv = csvpath + '/5fcv_vld.csv'
        traincsv = csvpath + '/5fcv_train.csv'

        for i, v in enumerate(datasets):
            # use 3 for train, 1 for vld
            v.to_csv(vldcsv, columns=['idx', 'CGI_seq', 'Label'], index=0)
            tempt = pd.DataFrame(columns=['idx', 'CGI_seq', 'Label'])
            for j, m in enumerate(datasets):
                if i != j:
                    tempt = tempt.append(m)
            tempt.to_csv(traincsv, columns=['idx', 'CGI_seq', 'Label'], index=0)

            # start training, validating and testing
            losses, train_acc, train_pre, train_recall, train_F, \
            vld_losses, vld_acc, vld_pre, vld_recall, vld_F, vo = \
                starttrain(allcgikmercsv, traincsv, vldcsv, modelpath, args.D, args.epoch, args.H, args.dp, args.lr,
                           csvpath, args.decay, args.filetitle, kind)
            # record loss and result(accuary,precsion,recall,F-value) of training, validation and test
            all_trian_loss.append(losses)
            all_train_acc.append(train_acc)
            all_train_pre.append(train_pre)
            all_train_recall.append(train_recall)
            all_train_F.append(train_F)
            all_vld_loss.append(vld_losses)
            all_vld_acc.append(vld_acc)
            all_vld_pred.append(vld_pre)
            all_vld_recall.append(vld_recall)
            all_vld_F.append(vld_F)

        print('-----------record-------------')

        # caculate average of test accuary,precsion,recall,F-value
        final_vld_acc = final(all_vld_acc, args.f, args.epoch)
        final_vld_pred = final(all_vld_pred, args.f, args.epoch)
        final_vld_recall = final(all_vld_recall, args.f, args.epoch)
        final_vld_F = final(all_vld_F, args.f, args.epoch)

        vld_avg_acc = average(final_vld_acc)
        vld_avg_pre = average(final_vld_pred)
        vld_avg_re = average(final_vld_recall)
        vld_avg_F = average(final_vld_F)

        # caculate standard loss of test accuary,precsion,recall,F-value
        vld_sl_acc = standardloss(final_vld_acc)
        vld_sl_pre = standardloss(final_vld_pred)
        vld_sl_re = standardloss(final_vld_recall)
        vld_sl_F = standardloss(final_vld_F)

        # record in csv file
        result = result.append(pd.DataFrame(
            {'type': [kind], 'vo': [vo], 'embedding_vector_size': [args.D], 'neighbour_winodw': [args.w],
             'epoch': [args.epoch],
             'hidden_nodes': [args.H], 'dropout': [args.dp], 'learning_ratio': [args.lr], 'weight_decay': [args.decay],
             'train_loss': [all_trian_loss],
             'train_accuary': [all_train_acc], 'train_precision': [all_train_pre],
             'train_recall': [all_train_recall], 'train_F-value': [all_train_F],
             'validate_loss': [all_vld_loss],
             'validate_accuary': [all_vld_acc], 'validate_precision': [all_vld_pred],
             'validate_recall': [all_vld_recall], 'validate_F-value': [all_vld_F],
             'final_vld_accuary': [final_vld_acc], 'final_vld_precision': [final_vld_pred],
             'final_vld_recall': [final_vld_recall],
             'final_vld_F-value': [final_vld_F],
             'average_vld_accuary': [vld_avg_acc], 'average_vld_precision': [vld_avg_pre],
             'average_vld_recall': [vld_avg_re], 'average_vld_F-value': [vld_avg_F],
             'sl_acc': [vld_sl_acc], 'sl_prec': [vld_sl_pre], 'sl_re': [vld_sl_re],
             'sl_F': [vld_sl_F]}),
            ignore_index=True)

        # resultcsvpath = rootpath + args.filetitle + '.csv'
        resultcsvpath = os.path.join(rootpath, args.filetitle + '.csv')

        result.to_csv(resultcsvpath,
                      columns=(
                          'type', 'vo', 'embedding_vector_size', 'neighbour_winodw', 'epoch', 'hidden_nodes', 'dropout',
                          'learning_ratio', 'weight_decay',
                          'train_loss',
                          'train_accuary', 'train_precision', 'train_recall', 'train_F-value',
                          'validate_loss',
                          'validate_accuary', 'validate_precision', 'validate_recall', 'validate_F-value',
                          'final_vld_accuary', 'final_vld_precision', 'final_vld_recall', 'final_vld_F-value',
                          'average_vld_accuary', 'average_vld_precision', 'average_vld_recall', 'average_vld_F-value',
                          'sl_acc', 'sl_prec', 'sl_re', 'sl_F'), index=0)
        result = pd.read_csv(resultcsvpath)
        torch.cuda.empty_cache()

