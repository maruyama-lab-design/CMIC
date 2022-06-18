import os
import pandas as pd
import argparse

import cmic2

top_data_dir = os.path.join("/home", "om", "OneDrive",
    "work", "methyl", "source_data", "cgi_methyl_fgo_blastocyst-maternal_unmethyl-pos")

def run(tag, top_data_dir=top_data_dir, embedding_freeze=True, shuffle_wv=False, debug_mode=False):


    input_data_dir = os.path.join(top_data_dir, tag)

    output_filename_prefix = os.path.join(top_data_dir, tag)
    if debug_mode:
        output_filename_prefix += "_debug"

    df_all_output = pd.DataFrame()
    for index, sub_dir in enumerate(os.listdir(input_data_dir)):
        print(f'\n{index}: {sub_dir}')
        data_dir = os.path.join(input_data_dir, sub_dir)
        df_output= cmic2.execute_KFoldCV(data_dir, exp_type_label=sub_dir, 
            embedding_freeze=embedding_freeze, shuffle_wv=shuffle_wv, debug_mode=debug_mode)

        if df_all_output.empty:
            df_all_output = df_output
        else:
            df_all_output = pd.concat([df_all_output, df_output])

        df_all_output.to_csv(output_filename_prefix + "_" + str(index) + ".csv")

    df_all_output.to_csv(output_filename_prefix + ".csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### To-do: replace the next with __main__ part of cmic2.py 

    # tag = "word2vec_cmic_kmin_kmax"
    # tag = "embedding_vec_dim"
    parser.add_argument('tag', help="the directory name directly having input sets like 'k3_7_aug1000_v10_w20_e5_a0.025_mina0.0001_splitDNA2vec'")

    parser.add_argument('-d', '--top_data_dir', help='top data directory', type=str, default=top_data_dir)
    parser.add_argument('-e', '--embedding_freeze', type=bool, default=True)
    parser.add_argument('-s', '--shuffle_wv', type=bool, default=False)
    parser.add_argument('-m', '--debug_mode', type=bool, default=False)
    args = parser.parse_args()
    run(args.tag, 
        top_data_dir=args.top_data_dir, 
        embedding_freeze=args.embedding_freeze, 
        shuffle_wv=args.shuffle_wv,
        debug_mode=args.debug_mode)
