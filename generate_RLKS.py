import argparse
import os
import random
from itertools import islice

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from splitDNA2vec import read_seq_file

'''
Generate Random Length k-mer Sequence to csv file.

INPUT: m.fasta & u.fasta that are generated in methyl lib.
OUTPUT: RLKS.csv that includes sequence index, RLKS, and its label.
The same seq_idx each have the same sequence, but the k-mer is different.
'''


# same function as `generate_random_kmer_seq_1` and `random_chunks` in splitDNA2vec.py
def split_seq2kmer(seq, k_min, k_max):
    '''Split sequence to random length k-mer that has k_min ≦ k-mer ≦ k_max'''
    kmer_list = []
    while len(seq) >= k_min:
        randint = random.randint(k_min, k_max)
        kmer_list.append(seq[:randint])
        seq = seq[randint:]
    return kmer_list


def generate_RLKS(idx, seq, k_min, k_max, aug_num, label):
    '''Augment a random length k-mer sequence to aug_num and returns a dataframe'''
    df_RLKS = pd.DataFrame(columns=['seq_idx', 'seq', 'label'])
    for n in range(aug_num):
        kmer_list = split_seq2kmer(seq, k_min, k_max)
        df_tmp = pd.DataFrame({'seq_idx': [idx], 'seq': [kmer_list], 'label': [label]})
        df_RLKS = df_RLKS.append(df_tmp)
    return df_RLKS


def add_df_to_csv(csv_path, df):
    df.to_csv(csv_path, columns=['seq_idx', 'seq', 'label'], mode='a', header=False, index=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_dir', help='path of fasta folder, default=\'./data/fasta\'',
                        default='./data/fasta')
    parser.add_argument('--out_dir', help='path of output folder, default=\'./data/RLKS\'',
                        default='./data/RLKS')
    parser.add_argument('--k_min', type=int,
                        help='minimum length of generated k-mer, default=3', default=3)
    parser.add_argument('--k_max', type=int,
                        help='maximum length of generated k-mer, default=6', default=6)
    parser.add_argument('--aug_num', type=int,
                        help='number of augment data, default=1000', default=1000)

    args = parser.parse_args()

    M2M_path = os.path.join(args.fasta_dir, 'm.fasta')
    M2U_path = os.path.join(args.fasta_dir, 'u.fasta')

    k_min = args.k_min
    k_max = args.k_max
    aug_num = args.aug_num
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_csv = os.path.join(out_dir, 'RLKS.csv')

    M2M_seq = read_seq_file(M2M_path)
    M2U_seq = read_seq_file(M2U_path)

    m_label = 1
    u_label = 0
    seq_idx = 0

    total_seq_num = len(M2M_seq)+len(M2U_seq)
    print('Extending {} sequence to {} random length k-mer sequence each...'.format(total_seq_num, aug_num))

    # for progress bar
    bar = tqdm(total=total_seq_num)

    pd.DataFrame(columns=['seq_idx', 'seq', 'label']).to_csv(out_csv, index=0)  # ヘッダーの作成
    for m_seq in M2M_seq:
        df_m = generate_RLKS(seq_idx, m_seq, k_min, k_max, aug_num, m_label)
        add_df_to_csv(out_csv, df_m)
        seq_idx += 1
        bar.update(1)

    for u_seq in M2U_seq:
        df_u = generate_RLKS(seq_idx, u_seq, k_min, k_max, aug_num, u_label)
        add_df_to_csv(out_csv, df_u)
        seq_idx += 1
        bar.update(1)
