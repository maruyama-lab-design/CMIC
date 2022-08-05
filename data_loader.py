import os
import numpy as np
import pandas as pd

# from CGI2meth.py -> '__main__'

# yaml file, config.yaml is not loaded. 

def load_Bock_dataset(Bock_dataset_filename="dataset_bock.csv"):
    # Bock_data_file = os.path.join("sample_data", "dataset_bock.csv")
    dir_name = os.path.dirname(Bock_dataset_filename)

    df = pd.read_csv(Bock_dataset_filename)
    pos_seq_list = df[df['Target attribute'].isin(['fully methylated'])].iloc[:, 5].values.tolist()
    neg_seq_list = df[df['Target attribute'].isin(['unmethylated'])].iloc[:, 5].values.tolist()
    print(f'pos = {len(pos_seq_list)}') # 29
    print(f'neg = {len(neg_seq_list)}') # 103

    seq = pos_seq_list + neg_seq_list
    seq_idx = list(range(len(pos_seq_list + neg_seq_list)))
    label = [1 for i in pos_seq_list] + [0 for j in neg_seq_list]
    # df_all_seq = pd.DataFrame({'seq_idx': seq_idx, 'seq': seq, 'label': label})
    df = pd.DataFrame({'seq_idx': seq_idx, 'seq': seq, 'label': label})
    return df, dir_name

def load_dataset(input_filename, M_lower_bound, U_upper_bound):
    df = pd.read_csv(input_filename)
    df['label'] = -1
    # 2022-04-23. 
    # Unmethylated and methylated CGIs are labeled as 1 and 0. 
    df.loc[df['beta_2'] >= M_lower_bound, 'label'] = 0  # 1
    df.loc[df['beta_2'] < U_upper_bound, 'label'] = 1   # 0
    print(sum(df['label']==1))
    print(sum(df['label']==0))
    print(sum(df['label']==-1))
    df = df[df.label != -1]

    if 'chrom' in df.columns and 'startpos' in df.columns:
        df = df.sort_values(['label', 'chrom', 'startpos'], ascending=[False, True, True])
    else:
        df = df.sort_values(['label'], ascending=[False])
    seq_idx = list(range(len(df)))
    df['seq_idx'] = seq_idx
    df = df.reset_index(drop=True)
    return df

if __name__ == "__main__":
    load_Bock_dataset(os.path.join("sample_data", "dataset_bock.csv"))