# from curses import A_ALTCHARSET
import os
import argparse

import torch 
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import gensim
import random
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import scipy.stats

import copy

# The next can be removed. 
# metric_names = ["loss", "balancedAccuracy", "precision", "recall", "F", "MCC", "AUC"]

class KmerSeqsDataset(Dataset):
    def __init__(self, input_tensor, input_label):
        self.X = input_tensor
        self.y = input_label

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # print(index)
        return [self.X[index], self.y[index]]

def collate_fn(batch):
    X, y = list(zip(*batch)) #[todo]numpy array
    X = [torch.tensor(x) for x in X]
    pad_X = pad_sequence(X, batch_first=True)
    y = torch.tensor(y, dtype=torch.float32)
    return pad_X, y


def makeDataLoader(instance_indexes, amp, input_data, wv, batch_size, collate_fn, 
    shuffle=True):
    row_indexes = np.concatenate(
        [i * amp + np.arange(amp, dtype=int) for i in instance_indexes]
        )

    yy = input_data['label'].iloc[row_indexes].values
    # yy = np.concatenate([yy, yy])

    list_of_kmer_sequences = [seq.split(' ') for seq in input_data['seq'].iloc[row_indexes] ]

    list_of_indexed_kmer_sequences = [ [wv.key_to_index[kmer]  for kmer in kmer_sequence] 
        for kmer_sequence in list_of_kmer_sequences]
    # reverse complement. 
    #   [  list(reversed([[str(Bio.Seq.Seq(kmer).reverse_complement())] for kmer in kmer_seq])) for kmer_seq in X]
    #
    # 2022-03-21
    # The next is commmented out because the reverse complements have already prepared. 
    # list_of_indexed_kmer_sequences_rev_comp = \
    #     [list(reversed(       [wv.key_to_index[str(Bio.Seq.Seq(kmer).reverse_complement())] \
    #         for kmer in kmer_sequence]    )) \
    #             for kmer_sequence in list_of_kmer_sequences]

    X = list_of_indexed_kmer_sequences # + list_of_indexed_kmer_sequences_rev_comp

    data_loader = DataLoader(KmerSeqsDataset(X,yy), batch_size, shuffle=shuffle, collate_fn=collate_fn)
    # dataloader.append(data_loader)
    return data_loader

class Classifier(nn.Module):
    def __init__(self, wv, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, embedding_freeze):
        super().__init__()
        """wv is supposed to hold vectors, key_to_index, and index_to_key."""
        self.wv = wv 
        word_vectors = torch.tensor(self.wv.vectors)

        if type(word_vectors) is not torch.Tensor:
            word_vectors = torch.tensor(word_vectors)

        self.embedding = nn.Embedding.from_pretrained(word_vectors, freeze=embedding_freeze) 
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional, 
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # fully connect layer
        self.act = nn.Sigmoid() # activate function

    def forward(self, index_seq):
        embedded = self.embedding(index_seq)

        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        # packed_output, hidden = self.gru(packed_embedded)
        packed_output, hidden = self.gru(embedded)

        # Concatenation of the final forward and backward hidden states. 
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 

        dense_outputs = self.fc(hidden) # Fully connect. 
        outputs = self.act(dense_outputs)
        return outputs

class ClassifierRNNType(nn.Module):
    def __init__(self, wv, embedding_dim, hidden_dim, output_dim, n_layers,
                 rnn_type, dropout, embedding_freeze):
        super().__init__()
        """wv is supposed to hold vectors, key_to_index, and index_to_key."""
        self.wv = wv 
        word_vectors = torch.tensor(self.wv.vectors)

        if type(word_vectors) is not torch.Tensor:
            word_vectors = torch.tensor(word_vectors)

        self.embedding = nn.Embedding.from_pretrained(word_vectors, freeze=embedding_freeze) 

        self.bidirectional = False
        if rnn_type[:2] == "Bi":
            self.bidirectional = True
        if rnn_type.endswith("RNN"):
            self.RNN_model = "RNN"
            self.rnn_type = nn.RNN(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=self.bidirectional, 
                            dropout=dropout,
                            batch_first=True)
        elif rnn_type.endswith("LSTM"):
            self.RNN_model = "LSTM"
            self.rnn_type = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=self.bidirectional, 
                            dropout=dropout,
                            batch_first=True)
        elif rnn_type.endswith("GRU"):
            self.RNN_model = "GRU"
            self.rnn_type = nn.GRU(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=self.bidirectional, 
                            dropout=dropout,
                            batch_first=True)
        else:
            self.RNN_model = "GRU"
            self.rnn_type = nn.GRU(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=self.bidirectional, 
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim) # fully connect layer
        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim) 

        self.act = nn.Sigmoid() # activate function

    def forward(self, index_seq):
        embedded = self.embedding(index_seq)

        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        # packed_output, hidden = self.gru(packed_embedded)

        if self.RNN_model == "LSTM":
            packed_output, (hidden, c_n) = self.rnn_type(embedded)
        else:
            packed_output, hidden = self.rnn_type(embedded)

        # Concatenation of the final forward and backward hidden states. 
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 

        dense_outputs = self.fc(hidden) # Fully connect. 
        outputs = self.act(dense_outputs)
        return outputs




def batch_weight(y):
    weight = y.clone().detach()
    pos_num = sum(y)
    neg_num = len(y) - pos_num
    weight[y==1] = neg_num / len(y)
    weight[y==0] = pos_num / len(y)
    return weight


def train(model, iterator, optimizer, device):
    print('training...')
    model.train() 
    
    y_trues = np.empty(0)
    y_preds = np.empty(0)
    all_loss = 0
    print(f"{len(iterator)} batches:", end=" ")
    for (index, batch) in enumerate(iterator):
        if index % 100 == 0:
            print(index, end=" ")
        
        # text, text_lengths = batch.CGI_seq # get text and the number of words
        # predictions = model(text, text_lengths).squeeze()         # convert to 1D tensor

        X = batch[0].to(device)
        y = batch[1].to(device)
        y_pred = model(X).squeeze()

        weight = batch_weight(y)
        loss_fn = nn.BCELoss(weight=weight).to(device)

        # loss = criterion(predictions, batch.Label) 
        loss = loss_fn(y_pred, y) 

        optimizer.zero_grad() # resets the gradients
        loss.backward()  # The loss is backpropaged, and the gradients are computed. 
        optimizer.step() # the weights are updated. 

        # acc, pred, recall, F, FPR = performance(predictions, batch.Label)
        # acc, pred, recall, F, FPR = performance(y_pred, y)

        # y_pred = np.condatenate([y_pred, y])

        all_loss += loss.item()
        y_trues = np.concatenate([y_trues,              y.to('cpu').detach().numpy().copy()])
        y_preds = np.concatenate([y_preds, y_pred.round().to('cpu').detach().numpy().copy()])

    print('')
    num_batches = len(iterator)
    all_loss /= num_batches
    acc = metrics.balanced_accuracy_score(y_trues, y_preds)
    pre = metrics.precision_score(y_trues, y_preds)
    rec = metrics.recall_score(y_trues, y_preds)
    f = metrics.f1_score(y_trues, y_preds)
    mcc = metrics.matthews_corrcoef(y_trues, y_preds)
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    auc = metrics.auc(fpr, tpr)
    cm = metrics.confusion_matrix(y_trues, y_preds)
    return all_loss, acc, pre, rec, f, mcc, auc, cm

# def test(model, iterator, loss_fn):
def test(model, iterator, device):
    # deactivating dropout layer
    print('testing...')
    model.eval()
    # loss, correct = 0, 0
    all_loss = 0

    y_trues = np.empty(0)
    y_preds = np.empty(0)
    with torch.no_grad():
        for batch in iterator:
            X = batch[0].to(device)
            y = batch[1].to(device)

            weight = batch_weight(y)
            loss_fn = nn.BCELoss(weight=weight).to(device)

            pred = model(X).squeeze()
            
            all_loss += loss_fn(pred, y).item()

            y_trues = np.concatenate([y_trues,            y.to('cpu').detach().numpy().copy()])
            y_preds = np.concatenate([y_preds, pred.round().to('cpu').detach().numpy().copy()])

            
            # 2021-12-27 次の役目が分からないので一旦comment out. 
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # loss /= num_batches
    # correct /= size
    num_batches = len(iterator)
    all_loss /= num_batches
    acc = metrics.balanced_accuracy_score(y_trues, y_preds)
    pre = metrics.precision_score(y_trues, y_preds)
    rec = metrics.recall_score(y_trues, y_preds)
    f = metrics.f1_score(y_trues, y_preds)
    mcc = metrics.matthews_corrcoef(y_trues, y_preds)
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    auc = metrics.auc(fpr, tpr)
    cm = metrics.confusion_matrix(y_trues, y_preds)
    return all_loss, acc, pre, rec, f, mcc, auc, cm


class RandomLengthKmerSeqDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, input_label):
        self.tensor = input_tensor
        self.label = input_label

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, index):
        return self.tensor[index], self.label[index]


def kmer_seq_multiplicity(input_data):
    num_of_CGIs = len(np.unique(input_data.seq_idx.values)) 
    num_of_rows = len(input_data)
    amp = num_of_rows // num_of_CGIs
    return amp

def select_short_inputs(input_data, max_length, amp):
    if "org_seq_len" in input_data.columns:
        input_data_short, input_data_long = select_short_inputs_with_org_seq_len(input_data, max_length)
    else:
        input_data_short, input_data_long = select_short_inputs_without_org_seq_len(input_data, max_length, amp)
    return input_data_short, input_data_long

def select_short_inputs_with_org_seq_len(input_data, max_length):
    org_seq_len = input_data['org_seq_len']
    is_less_than_equal_to = lambda x: x <= max_length
    valid = org_seq_len.map(is_less_than_equal_to)

    input_data['valid'] = valid
    input_data_short = input_data.query("valid == True")
    input_data_short.reset_index()
    input_data_long = input_data.query("valid == False")
    input_data_long.reset_index()  
    return input_data_short, input_data_long
    


def select_short_inputs_without_org_seq_len(input_data, max_length, amp):
    """This function does not work when k-mer stride is not 0 like dna2vec."""
    num_of_rows = len(input_data)

    valid = np.zeros(num_of_rows, dtype='int')
    for index, row in enumerate(input_data.itertuples()):
        if len(row.seq.replace(' ', '')) <= max_length: 
            valid[index] = True
        else:
            valid[index] = False

    # There are rows for the same CGI such that the values of valid are True and False. 
    for index in range(0, num_of_rows, amp):
        if False in valid[index:index+amp]:
            valid[index:index+amp] = False

    input_data['valid'] = valid
    input_data_short = input_data.query("valid == True")
    input_data_short.reset_index()
    input_data_long = input_data.query("valid == False")
    input_data_long.reset_index()  
    return input_data_short, input_data_long


def make_pos_neg_equal_sized(input_data, amp):
    num_of_pos = len(input_data.query("label == True")) // amp
    num_of_neg = len(input_data.query("label == False")) // amp
    min_num = min([num_of_pos, num_of_neg])

    # print(input_data.iloc[0:min_num*amp, 2])
    # print(input_data.iloc[num_of_pos*amp:(num_of_pos + min_num)*amp, 2])
    
    indexing = np.zeros(len(input_data), dtype='bool')
    # It is supposed that the first block is positive and the latter are negative.
    indexing[0:min_num*amp] = True
    indexing[num_of_pos*amp:(num_of_pos + min_num)*amp] = True
    input_data = input_data[indexing]

    input_data.reset_index()
    return input_data

class EpochWiseLog:
    def __init__(self):
        self.dict_array = []

    def add(self, fold_index, epoch_index, train_or_test, metric_names, metrics):
        row = {}
        row["Time"] = pd.to_datetime(datetime.datetime.today())
        row["Fold"] = fold_index
        row["Epoch"] = epoch_index
        row["Train|Test"] = train_or_test

        for index, key in enumerate(metric_names):
            row[key] = metrics[index]

        tn, fp, fn, tp = metrics[-1].flatten()
        row["TN"] = tn
        row["FP"] = fp
        row["FN"] = fn
        row["TP"] = tp

        # self.df = self.df.append(row, ignore_index=True)
        self.dict_array.append(row)
        

    def to_csv(self, filename, metric_names):
        df = pd.DataFrame(columns=["Time", "Fold", "Epoch", "Train|Test"] + metric_names)
        df = pd.concat([df, pd.DataFrame.from_dict(self.dict_array)])
        df.to_csv(filename, mode='a')


class OutputDataFrame:
    metric_names = ["loss", "balancedAccuracy", "precision", "recall", "F", "MCC", "AUC"]
    cv_data_types = ["train", "test"]
    contents = ["fold", "mean", "SE"]

    cols = ["type", "embedding_vector_size", "epoch", "hidden_nodes", "dropout", "learning_ratio", "weight_decay"]
    
    def __init__(self):
        tmp_cols = copy.copy(self.cols)
        for cv_data_type in self.cv_data_types:
            for content in self.contents:
                for met in self.metric_names:
                    new_col_name = "_".join([cv_data_type, met, content])
                    tmp_cols.append(new_col_name)
        self.df = pd.DataFrame(columns=tmp_cols)

        # self.df_output_row_index = 0

    def set(self, key, value, row_index="default_row"):
        # self.df_output.at[self.df_output_row_index, key] = value
        self.df.at[row_index, key] = value

    def record(self, metrics, row_index="default_row"):
        for cv_data_type_idx, cv_data_type in enumerate(self.cv_data_types): # training and test. 
            means = [ np.mean(metrics[cv_data_type_idx,:,_ii]) for _ii in range(metrics.shape[2]) ] # average metrics 
            stderrs = [ scipy.stats.sem(metrics[cv_data_type_idx,:,_ii]) for _ii in range(metrics.shape[2]) ] # standard errors

            for met_idx, met in enumerate(self.metric_names):         
                content  = self.contents[0]
                col_name = "_".join([cv_data_type, met, content])

                # error を出している．2022/05/17.
                print(metrics[cv_data_type_idx, :, met_idx])
                self.df.at[row_index, col_name] = metrics[cv_data_type_idx, :, met_idx]

                content  = self.contents[1]
                col_name = "_".join([cv_data_type, met, content])
                self.df.at[row_index, col_name] = means[met_idx]

                content  = self.contents[2]
                col_name = "_".join([cv_data_type, met, content])
                self.df.at[row_index, col_name] = stderrs[met_idx]
           
        return self.df



def execute_KFoldCV(data_dir, exp_type_label="NA", seq_max_len=500, RNN_type="BiGRU", embedding_freeze=True, 
    shuffle_wv=False, debug_mode=False, test_long=False):

    """
    To-Do: 
    Get options for max_length, and pos-neg-set-size-equality. 
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = {}".format(device))

    input_data = pd.read_csv(os.path.join(data_dir, "input_data.csv"))
    amp = kmer_seq_multiplicity(input_data) # amp is used when y is generated. 

    # seq_max_len = 500
    input_data, input_data_long = select_short_inputs(input_data, seq_max_len, amp)
    # input_data = make_pos_neg_equal_sized(input_data, amp)
    
    # The next can be move down to member functions, set. 
    output_df = OutputDataFrame()

    ### load a word2vec model
    word2vec_model = gensim.models.Word2Vec.load(os.path.join(data_dir, "word2vec.model"))
    wv = word2vec_model.wv

    num_output_nodes = 1
    n_layers = 1
    bidirectional = True

    embedding_dim = wv.vector_size 
    num_hidden_nodes = 256
    epoch = 2
    lr = 10e-4 
    batch_size = 32
    dropout = 0.5
    decay = 0.01

    n_splits = 3 # 3-fold cross-validation.

    if shuffle_wv: 
        # The next function, numpy.random.shuffle, updates wv.vectors itself, and returns no output.
        np.random.shuffle(wv.vectors)

    if debug_mode:
        epoch = 2
        n_splits = 2

    output_df.set("type", exp_type_label)
    output_df.set("embedding_vector_size", embedding_dim)
    output_df.set("epoch", epoch)
    output_df.set("hidden_nodes", num_hidden_nodes)
    output_df.set("dropout", dropout)
    output_df.set("learning_ratio", lr)
    output_df.set("weight_decay", decay)

    ### k-fold cross-validation related part. 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    num_of_CGIs = len(np.unique(input_data.seq_idx.values)) 
    X = np.array(range(num_of_CGIs))  
    y = np.array(input_data.label[0::amp]) # amp has been given above. 

    print(f"The number of CGIs used in a c.v. is {num_of_CGIs}.")
    print(f"The number of positive CGIs is {sum(y)}.")
    print(f"The number of long CGIs is {len(input_data_long)}.")

    # Save epoch-wise log.
    epoch_wise_log = EpochWiseLog()



    if test_long:
        models = []
        for fold_index in range(n_splits):
            if RNN_type == "BiGRU":
                model = Classifier(wv, embedding_dim, num_hidden_nodes, num_output_nodes, n_layers,
                        bidirectional, dropout, embedding_freeze).to(device)
            else:
                model = ClassifierRNNType(wv, embedding_dim, num_hidden_nodes, num_output_nodes, n_layers,
                        RNN_type, dropout, embedding_freeze).to(device)
            model.load_state_dict(torch.load(os.path.join(data_dir, "trained_" + str(fold_index) + ".model")))
            models.append(model)

        len_step = 100
        min_lengths = range(seq_max_len+1, 1002, len_step) # 6 sections. 
        for length_range_index, Lmin in enumerate(min_lengths):
            print(f'minimum length={Lmin}')
            Lmax = Lmin + len_step
            if length_range_index == len(min_lengths) -1:
                output_df.set("type", str(Lmin) + '<=', row_index=length_range_index)
            else:
                output_df.set("type", '[' + str(Lmin) + ', ' + str(Lmax) + ']', row_index=length_range_index)

            instance_indexes = []
            for cgi_level_index, amp_index in enumerate(range(0, len(input_data_long), amp)):
                cgi_len = input_data_long['org_seq_len'].iloc[amp_index]
                if length_range_index == len(min_lengths) -1:
                    if Lmin <= cgi_len:
                        instance_indexes.append(cgi_level_index)
                else:
                    if Lmin <= cgi_len < Lmax:
                        instance_indexes.append(cgi_level_index)
            dataloader = makeDataLoader(instance_indexes, amp, input_data_long, wv, batch_size, collate_fn, shuffle=False)
        
            metrics = np.zeros((2, len(models), len(OutputDataFrame.metric_names))) 
            for model_index, model in enumerate(models):
                print(f'model index={model_index}')

                test_metrics = test(model, dataloader, device)
                print('Test: loss={}, balanced_accuracy={}, precision={}, recall={}, F={}, MCC={}, AUC={}'.format(
                    test_metrics[0],
                    test_metrics[1],
                    test_metrics[2],
                    test_metrics[3],
                    test_metrics[4],
                    test_metrics[5],
                    test_metrics[6]))
                metrics[1, model_index] = test_metrics[:-1] # The last element, confusion matrix, is excluded. 
            output_df.record(metrics, row_index=length_range_index)    
        return output_df.df

        


    #(training and test), fold number, and number of metrics
    metrics = np.zeros((2, n_splits, len(OutputDataFrame.metric_names))) 
    # models = [] # keep trained models. 

    for fold_index, (train_index, test_index) in enumerate(skf.split(X, y)):
        if RNN_type == "BiGRU":
            model = Classifier(wv, embedding_dim, num_hidden_nodes, num_output_nodes, n_layers,
                        bidirectional, dropout, embedding_freeze).to(device)
        else:
            model = ClassifierRNNType(wv, embedding_dim, num_hidden_nodes, num_output_nodes, n_layers,
                    RNN_type, dropout, embedding_freeze).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        dataloaders = []
        for instance_indexes in [train_index, test_index]: # These indexes are assigned to CGIs, not augmented k-mer sequences. 
            dl = makeDataLoader(instance_indexes, amp, input_data, wv, batch_size, collate_fn)
            dataloaders.append(dl)
        
        for epoch_index in range(epoch):
            print(f"Fold: {fold_index + 1}/{n_splits}, Epoch: {epoch_index + 1}/{epoch}")
            train_metrics = train(model, dataloaders[0], optimizer, device)
            print('Training: loss={}, balanced_accuracy={}, precision={}, recall={}, F={}, MCC={}, AUC={}'.format(
                train_metrics[0],
                train_metrics[1],
                train_metrics[2],
                train_metrics[3],
                train_metrics[4],
                train_metrics[5],
                train_metrics[6]))
            epoch_wise_log.add(fold_index, epoch_index, "train", output_df.metric_names, train_metrics)
           
            test_metrics = test(model, dataloaders[1], device)
            print('Test: loss={}, balanced_accuracy={}, precision={}, recall={}, F={}, MCC={}, AUC={}'.format(
                test_metrics[0],
                test_metrics[1],
                test_metrics[2],
                test_metrics[3],
                test_metrics[4],
                test_metrics[5],
                test_metrics[6]))
            epoch_wise_log.add(fold_index, epoch_index, "test", output_df.metric_names, test_metrics)

        # Here we get a test result of a fold. 
        metrics[0, fold_index] = train_metrics[:-1] # The last element, confusion matrix, is excluded. 
        metrics[1, fold_index] = test_metrics[:-1] 

        torch.save(model.state_dict(), 
            os.path.join(data_dir, "trained_" + str(fold_index) + ".model"))
        # models.append(model)
    epoch_wise_log_filename = os.path.join(data_dir, "epoch_wise_log.csv")
    epoch_wise_log.to_csv(epoch_wise_log_filename, output_df.metric_names)
    return output_df.record(metrics)
    # end of for-loop of k-fold CV. 


if __name__ == "__main__":
    # top_data_dir = os.path.join("/home", "om", "OneDrive", 
    # "work", "methyl", "source_data", "cgi_methyl_fgo_blastocyst-maternal_unmethyl-pos", 
    # "word2vec_embedding_vec_dim") 
    # data_dir = os.path.join(top_data_dir, 
    #     "k3_7_aug1000_v10_w20_e5_a0.025_mina0.0001_splitDNA2vec"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", help="This looks like exp_kmin_kmax/k4_11_aug1000_v20_w10_e10_a0.025_mina0.0001_splitDNA2vec")
    parser.add_argument('-o', '--output', help="specify the output file name", type=str, default="output.csv")
    parser.add_argument('--seq_max_len', help="The longer sequences are excluded.", type=int, default=500)
    parser.add_argument("-r", "--RNN_type", type=str, default="BiGRU", help="RNN, BiRNN, GRU, BiGRU, LSTM, BiLSTM")
    parser.add_argument('--embedding_freeze', type=bool, default=True, help="The embedding vectors are not updated.")
    parser.add_argument('--shuffle_wv', type=bool, default=False, help="The word2vec mapping from k-mers to embedding vecotrs are shuffled before initializing the embedding layer.")
    parser.add_argument('--debug_mode', type=bool, default=False, help="Obsolete.")
    parser.add_argument('--test_long', type=bool, default=False, help="Once trained models are obtained, this option is used to test longer CGIs.")
    args = parser.parse_args()  

    # df_output = execute_KFoldCV(input_data, word2vec_model.wv)
    df_output = execute_KFoldCV(args.data_dir, 
        exp_type_label = "NA", 
        seq_max_len = args.seq_max_len, 
        RNN_type=args.RNN_type, 
        embedding_freeze=args.embedding_freeze, 
        shuffle_wv=args.shuffle_wv, 
        debug_mode=args.debug_mode,
        test_long=args.test_long)
    output_filename = os.path.join(args.data_dir, args.output)
    df_output.to_csv(output_filename, mode='a', header=True)    


         
    