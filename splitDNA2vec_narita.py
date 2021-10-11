"""
Generate word2vec or doc2vec model from fasta file.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import LineSentence
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


# -----functions to split sequence for k-fold CV from fasta file-----
def get_df(path):
    return pd.read_csv(path, encoding="utf-8")


def read_seq_file(path):
    sequences = []
    for seq_record in SeqIO.parse(path, "fasta"):
        seq = str(seq_record.seq).upper() #  make all sequences uppercase
        sequences.append(seq)
    return sequences


def make_df_from_fasta(fasta_dir, needs_rc):
    '''
    Return a dataframe with column of seq_idx, seq, and label
    '''
    M2M_path = os.path.join(fasta_dir, 'm.fasta')
    M2U_path = os.path.join(fasta_dir, 'u.fasta')

    m_list = read_seq_file(M2M_path)
    u_list = read_seq_file(M2U_path)
    if needs_rc:
        rc_m_list = [str(Seq(s).reverse_complement()) for s in m_list]
        rc_u_list = [str(Seq(s).reverse_complement()) for s in u_list]
        seq = m_list + u_list + rc_m_list + rc_u_list
        seq_idx = list(range(len(m_list + u_list))) + list(range(len(rc_m_list + rc_u_list)))
        label = [1 for i in m_list] + [0 for j in u_list] + [1 for i in rc_m_list] + [0 for j in rc_u_list]
    else:
        seq = m_list + u_list
        seq_idx = list(range(len(m_list + u_list)))
        label = [1 for i in m_list] + [0 for j in u_list]

    return pd.DataFrame({'seq_idx': seq_idx, 'seq': seq, 'label': label})


def make_kfold_csv(df, split_num, out_dir):
    '''
    Split index from dataframe into split_num and make csv file for each to tmp directory
    '''
    skf = StratifiedKFold(n_splits=split_num, shuffle=True, random_state=0)
    set_k = 0
    # 0:index, 1:label, 2~:feature
    for train_index, test_index in skf.split(df.iloc[:, 2:], df['label']):
        train_index = train_index.tolist()
        test_index = test_index.tolist()

        output = os.path.join(out_dir, 'set_{}'.format(set_k))
        os.makedirs(output, exist_ok=True)

        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        df_train.to_csv(os.path.join(output, 'train.csv'), index=False)
        df_test.to_csv(os.path.join(output, 'test.csv'), index=False)

        set_k += 1



# -----functions to generate random length k-mer sequence-----
def split_seq2kmer(seq, k_min, k_max):
    '''
    Split sequence to random length k-mer that has k_min ≦ k-mer ≦ k_max
    '''
    kmer_list = []
    while len(seq) >= k_min:
        randint = random.randint(k_min, k_max)
        kmer_list.append(seq[:randint])
        seq = seq[randint:]
    return kmer_list


def generate_single_RLKS(idx, seq, label, k_min, k_max, aug_num):
    '''
    Augment a random length k-mer sequence to aug_num and returns a dataframe
    '''
    df_RLKS = pd.DataFrame(columns=['seq_idx', 'seq', 'label'])
    for n in range(aug_num):
        kmer_list = split_seq2kmer(seq, k_min, k_max)
        df_tmp = pd.DataFrame({'seq_idx': [idx], 'seq': [kmer_list], 'label': [label]})
        df_RLKS = df_RLKS.append(df_tmp)
    return df_RLKS


def generate_RLKS_to_csv(df, k_min, k_max, aug_num, out_csv):
    '''
    Make all sequences in dataframe into RLKS and write them to csv
    '''
    # make header
    pd.DataFrame(columns=['seq_idx', 'seq', 'label']).to_csv(out_csv, index=0)

    # NOTE: 1行ずつ扱うので，dfを使うよりもcsvに逐次書き込んだ方が処理が早い
    for seq_idx, seq, label in tqdm(zip(df['seq_idx'], df['seq'], df['label']), total=len(df), desc='[RLKS]'):
        df_RLKS = generate_single_RLKS(seq_idx, seq, label, k_min, k_max, aug_num)
        df_RLKS['seq'] = df_RLKS['seq'].apply(' '.join) # change seq type list to str
        df_RLKS.to_csv(out_csv, columns=['seq_idx', 'seq', 'label'], mode='a', header=False, index=0)



# -----functions to train for each set-----
def filter_RLKS(seq_idx, df_RLKS):
    '''
    Return RLKS dataframe filtered by the same index as seq_idx
    '''
    df_filtered = df_RLKS[df_RLKS['seq_idx'].isin(seq_idx)]
    return df_filtered


def train_model(data_dir, mode, min_count, vec_size, window):
    '''
    Train with the model selected by args. This saves model file to data_dir and returns model
    '''
    df_RLKS = get_df(os.path.join(data_dir, 'RLKS.csv'))

    if mode == 'word2vec':
        # compute model
        RLKS_str = '\n'.join(df_RLKS['seq'].values.tolist())
        RLKS_path = os.path.join(data_dir, 'w2v_RLKS.txt')
        with open(RLKS_path, 'w') as f:
            f.write(RLKS_str)
        model = train_w2v_model(RLKS_path, min_count, vec_size, window)
        model.save(data_dir + '/word2vec.model')
        model.wv.save_word2vec_format(data_dir + '/word2vec.txt', binary=False)

    if mode == 'doc2vec':
        # compute model
        df_RLKS['seq'] = df_RLKS['seq'].str.split(' ') # change seq type str to list
        tag_list = df_RLKS['seq_idx'].values.astype('str')
        text_list = df_RLKS['seq'].values
        documents = generate_d2v_doc(tag_list, text_list)
        model = train_d2v_model(documents, min_count, vec_size, window)
        model.save(data_dir + '/doc2vec.model')
        
    return model

# -----functions to train word2vec-----
def train_w2v_model(sentence_path, min_count, vec_size, window):
    model = Word2Vec(
        sentences=LineSentence(sentence_path),
        vector_size=vec_size,
        min_count=min_count,
        window=window,
        workers=4
    )
    return model



# -----functions to train doc2vec-----
def train_d2v_model(documents, min_count, vec_size, window):
    model = Doc2Vec(
        documents=documents,
        min_count=min_count,
        vector_size=vec_size,
        window=window,
        dm=1,
        epochs=20
    )
    return model


def generate_d2v_doc(tag_list, token_list):
    documents = []
    for tag, token in zip(tag_list, token_list):
        documents.append(TaggedDocument(tags=[tag], words=token))
    return documents


def concat_df_from_model(df, model, vec_size):
    '''
    Make dataframe that has column of below
    0: seq_idx, 1: label, 2~: features
    '''
    feature_columns = []
    [feature_columns.append('feature{}'.format(f_idx)) for f_idx in range(vec_size)]
    df = pd.DataFrame(columns=feature_columns)

    seq_idx = df['seq_idx']
    label = df['label']
    for idx in range(len(model.dv)):
        df.loc[idx] = model.dv[idx]
    df.insert(0, 'seq_idx', seq_idx)
    df.insert(1, 'label', label)
    return df



# -----main function-----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_dir', help='input path of directory which includes m.fasta and u.fasta')
    parser.add_argument('mode', help='specify the type of embedding vector whether word2vec or doc2vec')
    
    parser.add_argument('-o', '--out_dir', help='path of output directory. default=./data/out', default='./data/out')
    parser.add_argument('-s', '--split_num', type=int,help='split number to be used in k-fold cross validation, default=5', default=5)
    parser.add_argument('-kmin', '--k_min', type=int,help='minimum length of generated k-mer, default=3', default=3)
    parser.add_argument('-kmax', '--k_max', type=int,help='maximum length of generated k-mer, default=6', default=6)
    parser.add_argument('-rc', '--rc', type=bool,help='use reverse compliment or not, default=True', default=True)
    parser.add_argument('-aug', '--aug_num', type=int,help='number of augment data, default=1000', default=1000)

    parser.add_argument('-mcnt', '--min_count', type=int,help='minimum count option for model, default=1', default=1)
    parser.add_argument('-v', '--vec_size', type=int,help='size of feature vector that model has, default=300', default=300)
    parser.add_argument('-w', '--window', type=int,help='neighbour window for model, default=10', default=10)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # read fasta to dataframe
    df_all_seq = make_df_from_fasta(args.fasta_dir, args.rc)

    # generate RLKS.csv
    RLKS_path = os.path.join(args.out_dir, 'RLKS.csv')
    generate_RLKS_to_csv(df_all_seq, args.k_min, args.k_max, args.aug_num, RLKS_path)

    # run training
    model = train_model(args.out_dir, args.mode, args.min_count, args.vec_size, args.window)
    df_embed = concat_df_from_model(df_all_seq, model, args.vec_size)

    # preprocessing for k-fold CV
    make_kfold_csv(df_embed, args.split_num, args.out_dir)
