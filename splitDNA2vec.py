"""
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
# from Bio import SeqIO
from Bio.Seq import Seq
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import LineSentence
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import multiprocessing

import data_loader


# -----functions to split sequence for k-fold CV from fasta file-----
def read_seq_file(filepath):
    sequences = [str(seq_record.seq.upper()) for seq_record in SeqIO.parse(filepath, "fasta")]
    return sequences


def make_df_from_fasta(pos_fasta_path, neg_fasta_path):
    '''
    Return a dataframe with column of seq_idx, seq, and label
    '''
    pos_list = read_seq_file(pos_fasta_path)
    neg_list = read_seq_file(neg_fasta_path)
    seq = pos_list + neg_list
    seq_idx = list(range(len(pos_list + neg_list)))
    label = [1 for i in pos_list] + [0 for j in neg_list]

    ### seq_idx is used as CGI sequence IDs. 
    return pd.DataFrame({'seq_idx': seq_idx, 'seq': seq, 'label': label})


# -----functions to generate random length k-mer sequence-----
def split_k_mer_seq(seq, k, stride):
    start_pos_list = range(0, len(seq), stride)
    kmer_list = []
    for start_pos in start_pos_list:
        kmer = seq[start_pos: start_pos+k] 
        if len(kmer) == k:
            kmer_list.append(kmer)
    return kmer_list

def repeat_split_k_mer_seq(idx, seq, label, k_and_stride_list, aug_num):
    '''
    For each length, k, in k_list, 
    a given sequence, seq, is splitted into k-mers with the given stride. 
    '''
    df = pd.DataFrame(columns=['seq_idx', 'seq', 'label', 'org_seq_len'])
    for (k, stride) in k_and_stride_list:
        kmer_list = split_k_mer_seq(seq, k, stride)
        df_tmp = pd.DataFrame({'seq_idx': [idx], 'seq': [kmer_list], 'label': [label], 
            'org_seq_len': [len(seq)]})
        # df = df.append(df_tmp) # deprecated 
        df = pd.concat([df, df_tmp])
    return df    


def split_random_kmin_kmax_mer_seq(seq, k_min, k_max, stride):
    '''
    Split a DNA sequence to variable-length k-mers whose lengths are ranged from k_min to k_max. 
    If stride is 0, this runs in splitDNA2vec mode. 
    If strinde is 1, this runs as dna2vec. 
    If k is shorter than stride, the value of stride is temporaly shrunk to k. 
    '''
    kmer_list = []
    while len(seq) >= k_min:
        randint = random.randint(k_min, k_max)
        kmer_list.append(seq[:randint])

        # if stride =< 0 then GO TO else
        # if stride => randint then GO TO else
        # Otherwise GO TO the next: 
        if 0 < stride < randint:
            seq = seq[stride:]
        else:
            seq = seq[randint:]
    return kmer_list

def repeat_split_random_kmin_kmax_mer_seq(idx, seq, label, k_min, k_max, stride, aug_num):
    '''
    Repeatedly run the above function, split_random_kmin_kmax_mer_seq. 
    '''
    df_RLKS = pd.DataFrame(columns=['seq_idx', 'seq', 'label', 'org_seq_len'])
    for n in range(aug_num):
        kmer_list = split_random_kmin_kmax_mer_seq(seq, k_min, k_max, stride)
        df_tmp = pd.DataFrame({'seq_idx': [idx], 'seq': [kmer_list], 'label': [label], 
            'org_seq_len': [len(seq)] })
        # df_RLKS = df_RLKS.append(df_tmp) # deprecated 
        df_RLKS = pd.concat([df_RLKS, df_tmp])
    return df_RLKS

def generate_input_data_to_csv(df, random_kmer, k_and_stride, aug_num, out_csv, needs_rc):
    '''
    Make all sequences in a dataframe into sequence of k-mers, and save the outcomes to a csv file. 
    This function is called by make below and another code, CGI2meth.py. 
    '''
    if random_kmer:
        k_min = k_and_stride[0]
        k_max = k_and_stride[1]
        stride = k_and_stride[2]
    else:
        # This is a list of pairs of (k,stride). 
        k_and_stride_list = k_and_stride

    # make header
    pd.DataFrame(columns=['seq_idx', 'seq', 'label', 'org_seq_len']).to_csv(out_csv, index=0)

    for seq_idx, seq, label in tqdm(zip(df['seq_idx'], df['seq'], df['label']), total=len(df), desc='[data prep]'):
        if random_kmer:
            df_RLKS = repeat_split_random_kmin_kmax_mer_seq(seq_idx, seq, label, k_min, k_max, stride, aug_num)
        else:
            df_RLKS = repeat_split_k_mer_seq(seq_idx, seq, label, k_and_stride_list, aug_num)

        df_RLKS['seq'] = df_RLKS['seq'].apply(' '.join) # change the seq type from list to str with separator ' '. 
        df_RLKS.to_csv(out_csv, columns=['seq_idx', 'seq', 'label', 'org_seq_len'], mode='a', header=False, index=0)

        if needs_rc:
            rc_seq = str(Seq(seq).reverse_complement())
            if random_kmer:
                df_rc_RLKS = repeat_split_random_kmin_kmax_mer_seq(seq_idx, rc_seq, label, k_min, k_max, stride, aug_num)
            else:
                df_rc_RLKS = repeat_split_k_mer_seq(seq_idx, rc_seq, label, k_and_stride_list, aug_num)

            df_rc_RLKS['seq'] = df_rc_RLKS['seq'].apply(' '.join) # change seq type list to str
            df_rc_RLKS.to_csv(out_csv, columns=['seq_idx', 'seq', 'label', 'org_seq_len'], mode='a', header=False, index=0)



# -----functions to train model-----
def train_model(data_dir, input_path, mode, shuffle_input, min_count, vec_size, window, epoch, alpha, min_alpha):
    '''
    Train with the model selected by args
    '''
    df_input = pd.read_csv(input_path, encoding="utf-8")

    # Randomly shuffle the order of the input rows of DataFrame, df_input. 
    if shuffle_input:
        df_input = df_input.sample(frac=1, random_state=0)
        df_input.to_csv(input_path, columns=['seq_idx', 'seq', 'label'], mode='w', index=0)

    cpu_num = multiprocessing.cpu_count()
    if mode == 'doc2vec':
        df_input['seq'] = df_input['seq'].str.split(' ') # change seq type str to list
        text_list = df_input['seq'].values
        tag_list = df_input['seq_idx'].values.astype('str')

        documents = generate_d2v_doc(tag_list, text_list)
        
        # model = train_d2v_model(documents, min_count, vec_size, window, epoch, alpha, min_alpha)
        print("running doc2vec...")
        model = Doc2Vec(
            min_count=min_count,
            vector_size=vec_size,
            window=window,
            dm=1,
            workers= int(cpu_num * 3/4), 
            epochs=epoch,
            alpha=alpha,
            min_alpha=min_alpha
        )
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(data_dir + '/doc2vec.model')


    if mode == 'word2vec':
        input_str = '\n'.join(df_input['seq'].values.tolist())
        input_path = os.path.join(data_dir, 'w2v_input.txt')
        with open(input_path, 'w') as f:
            f.write(input_str)
        
        print("running word2vec...")
        # model = train_w2v_model(input_path, min_count, vec_size, window, epoch, alpha, min_alpha)
        model = model = Word2Vec(sentences=LineSentence(input_path), vector_size=vec_size, 
            min_count=min_count,
            window=window,
            # workers=4,
            workers= int(cpu_num * 3/4), 
            epochs=epoch,
            alpha=alpha,
            min_alpha=min_alpha)

        model.save(data_dir + '/word2vec.model')
        model.wv.save_word2vec_format(data_dir + '/word2vec.txt', binary=False)
    
    return model


def generate_d2v_doc(tag_list, token_list):
    documents = []
    for tag, token in zip(tag_list, token_list):
        documents.append(TaggedDocument(tags=[tag], words=token))
    return documents

def make_d2v_df(df, model, vec_size):
    """
    The given DataFrame object, df, is transformed into 
    a new DataFrame object, df_d2v. 
    Their columns are as follows:
    df
    0: seq_idx, 1: seq, 2: label
    df_d2v
    0: seq_idx, 1: label, 2~: features

    This is used only with doc2vec, not word2vec. 
    """
    
    feature_columns = []
    [feature_columns.append('feature{}'.format(f_idx)) for f_idx in range(vec_size)]
    df_d2v = pd.DataFrame(columns=feature_columns)

    seq_idx = df['seq_idx'] # looks like 0, 2, 10, 14, ...
    label = df['label']

    for idx in seq_idx.index: # row names. 
        # doc2vec.dv has key_to_index, lile {'0':0, '2':1, '10':2, ... }. These values coincident with the values of seq_idx. 
        # idx of df_d2v.loc[idx] is not the index number of a row. That is an index name in DataFrame. 
        df_d2v.loc[idx] = model.dv[str(seq_idx[idx])] 

    df_d2v.insert(0, 'seq_idx', seq_idx)
    df_d2v.insert(1, 'label', label)
    return df_d2v


def make(args):
    work_name = 'k{}_{}_aug{}_v{}_w{}_e{}_a{}_mina{}_stride{}'.format(
        args["k_min"], 
        args["k_max"], 
        args["aug_num"], 
        args["vec_size"], 
        args["window"], 
        args["epoch"], 
        args["alpha"], 
        args["min_alpha"],
        args["stride"] )
    if args["shuffle_input"]:
        work_name += '_shuffle'
    working_dir = os.path.join(args["out_dir"], work_name)
    os.makedirs(working_dir, exist_ok=True)
    print(working_dir)
    labeled_data = os.path.join(working_dir, "input_data.csv")

    if "data_type" in args and args["data_type"] == "Bock_2006":
        print("Loading Bock_2006...")
        df_all_seq, dir_name = data_loader.load_Bock_dataset(args["input_file"])
    else:
        df_all_seq = data_loader.load_dataset(args["input_file"], args["M_lower_bound"], args["U_upper_bound"])

    # print(f'Before {len(df_all_seq)}')
    #
    # The next block has a bug that 
    # every row has the same sequence length
    # thoug they are different.
    # df_all_seq["seq_len"] = len(df_all_seq['seq'])
    # print(df_all_seq)
    # print(df_all_seq.columns)
    # df_all_seq = df_all_seq.query("seq_len >= 200")
    # 
    CGI_min_len = 200
    df_all_seq = df_all_seq.query("endpos - startpos >= @CGI_min_len")
    # print(f'After {len(df_all_seq)}')

    # generate input_data.csv using splitDNA2vec algorithm
    # Namely, the next code is specialized for extracting random (kmin, kmax)-mer sequences, 
    # not for generation sequences of constant length k-mer for specified multiple k values. 
    generate_input_data_to_csv(df_all_seq, 
        args["random_kmer"], 
        [args["k_min"], args["k_max"], args["stride"]], 
        args["aug_num"], labeled_data, args["rc"])

    # run training
    print('Training model...')

    model = train_model(working_dir, labeled_data, 
        args["mode"], 
        args["shuffle_input"], 
        args["min_count"], 
        args["vec_size"], 
        args["window"], 
        args["epoch"], 
        args["alpha"], 
        args["min_alpha"])

    # The next make a mapping from a DNA sequence ID to the corresponding paragraph vector. 
    if args["mode"] == 'doc2vec':
        df_embed = make_d2v_df(df_all_seq, model, args["vec_size"])
        df_embed.to_csv(os.path.join(working_dir, 'doc2vec.csv'), index=False)

if __name__ == '__main__':
    """
    This part is used for making input files for CMIC2,  
    and NOT used for other methods, repeat_CGI2meth.py and CGI2meth.py 
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 2022-04-07
    # parser.add_argument('pos_fasta_path', help='Path of a positive samples of FASTA file')
    # parser.add_argument('neg_fasta_path', help='Path of a negative samples of FASTA file')
    parser.add_argument('input_file', help='input csv file.')

    parser.add_argument('mode', choices=['doc2vec', 'word2vec'], help='Specify the type of embedding vector whether word2vec or doc2vec')
    
    # parser.add_argument('-input', '--input_path', help='Path of input data for model which is generated from FASTA fale. default=input_data.csv', default='input_data.csv')

    parser.add_argument('-o', '--out_dir', help='Path of output directory. default=output', default='output')

    parser.add_argument('-m', '--M_lower_bound', type=float, help='lower bound for methylation', default=0.4)
    parser.add_argument('-u', '--U_upper_bound', type=float, help='upper bound for unmethylation', default=0.1)

    parser.add_argument('-random_kmer', '--random_kmer', type=bool, help='If True, length k is randomly sampled from a uniform distribution of the interval [k_min, k_max], default=True for splitDNA2vec', default=True)
    parser.add_argument('-kmin', '--k_min', type=int,help='Minimum length of extracted k-mer', default=4)
    parser.add_argument('-kmax', '--k_max', type=int,help='Maximum length of extracted k-mer', default=12)

    parser.add_argument('--stride', type=int, help='k-mer extraction window stride. Set 0 for splitDNA2vec. Set 1 for dna2vec mode.', default=0)

    parser.add_argument('-rc', '--rc', type=bool,help='Include reverse compliments or not, default=True', default=True)
    parser.add_argument("-aug", '--aug_num', type=int, help='Number of k-mer sequences generated from the same DNA sequence, default=1000', default=1000)

    parser.add_argument('-mcnt', '--min_count', type=int,help='Ignores all words with total frequency lower than this.  default=1', default=1)
    parser.add_argument('-v', '--vec_size', type=int,help='Dimensionality of the word vectors', default=10)
    parser.add_argument('-w', '--window', type=int,help='Maximum distance between the current and predicted word within a sentence. default=10', default=10)
    parser.add_argument('-e', '--epoch', type=int, help='Number of iterations (epochs) of word2vec or doc2vec', default=10)
    parser.add_argument('-a', '--alpha', type=float, help='Initial learning rate for model', default=0.025)
    parser.add_argument('-ma', '--min_alpha', type=float, help='Learning rate will linearly drop to min_alpha as training progresses', default=0.0001)

    parser.add_argument("--data_type", type=str, help="Specify 'Bock_2006' when the Bock 2006 dataset is used", 
        default="not_specified")


    parser.add_argument('-shuffle', '--shuffle_input', type=bool,help='Randomly shuffle the order of the rows of input data before learning a model if True', default=False)

    args = vars(parser.parse_args())
    make(args)
