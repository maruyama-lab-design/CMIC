# How to run CMIC

1. splitDNA2vec

When you use your own CGI methylation data, 
a csv file is needed with headers, 
'beta_2' (target methylation rate), 
'seq' (DNA sequence), 
and optionally 
'chrom' (chromosome ID), 
'startpos' (start position). 
Our input dataset and Bock2006 dataset are in the data directory of this Github repository. 

The next command generates files, 
input_data.csv, 
w2v_input.txt, 
word2vec.model, and 
word2vec.txt, 
in a directory like, k4_12_aug1000_v20_w10_e10_a0.025_mina0.0001_stride0, 
under the specified or default directory. 

$python splitDNA2vec.py your_input_file.csv word2vec [options]

Among the output files, input_data.csv and word2vec.model are taken as input to CMIC2. 

usage: splitDNA2vec.py [-h] [-o OUT_DIR] [-m M_LOWER_BOUND] [-u U_UPPER_BOUND] [-random_kmer RANDOM_KMER] [-kmin K_MIN] [-kmax K_MAX] [--stride STRIDE] [-rc RC] [-aug AUG_NUM]
                       [-mcnt MIN_COUNT] [-v VEC_SIZE] [-w WINDOW] [-e EPOCH] [-a ALPHA] [-ma MIN_ALPHA] [--data_type DATA_TYPE] [-shuffle SHUFFLE_INPUT]
                       input_file {doc2vec,word2vec}

positional arguments:
  input_file            input csv file.
  {doc2vec,word2vec}    Specify the type of embedding vector whether word2vec or doc2vec

optional arguments:
  -h, --help            show this help message and exit

  -o OUT_DIR, --out_dir OUT_DIR
                        Path of output directory. default=output (default: output)

  -m M_LOWER_BOUND, --M_lower_bound M_LOWER_BOUND
                        lower bound for methylation (default: 0.4)

  -u U_UPPER_BOUND, --U_upper_bound U_UPPER_BOUND
                        upper bound for unmethylation (default: 0.1)

  -random_kmer RANDOM_KMER, --random_kmer RANDOM_KMER
                        If True, length k is randomly sampled from a uniform distribution of the interval [k_min, k_max], default=True for splitDNA2vec (default: True)

  -kmin K_MIN, --k_min K_MIN
                        Minimum length of extracted k-mer (default: 4)

  -kmax K_MAX, --k_max K_MAX
                        Maximum length of extracted k-mer (default: 12)

  --stride STRIDE       k-mer extraction window stride. Set 0 for splitDNA2vec. Set 1 for dna2vec mode. (default: 0)

  -rc RC, --rc RC       Include reverse compliments or not, default=True (default: True)

  -aug AUG_NUM, --aug_num AUG_NUM
                        Number of k-mer sequences generated from the same DNA sequence, default=1000 (default: 1000)

  -mcnt MIN_COUNT, --min_count MIN_COUNT
                        Ignores all words with total frequency lower than this. default=1 (default: 1)

  -v VEC_SIZE, --vec_size VEC_SIZE
                        Dimensionality of the word vectors (default: 10)

  -w WINDOW, --window WINDOW
                        Maximum distance between the current and predicted word within a sentence. default=10 (default: 10)

  -e EPOCH, --epoch EPOCH
                        Number of iterations (epochs) of word2vec or doc2vec (default: 10)

  -a ALPHA, --alpha ALPHA
                        Initial learning rate for model (default: 0.025)

  -ma MIN_ALPHA, --min_alpha MIN_ALPHA
                        Learning rate will linearly drop to min_alpha as training progresses (default: 0.0001)

  --data_type DATA_TYPE
                        Specify 'Bock_2006' when the Bock 2006 dataset is used (default: not_specified)

  -shuffle SHUFFLE_INPUT, --shuffle_input SHUFFLE_INPUT
                        Randomly shuffle the order of the rows of input data before learning a model if True (default: False)




2. CMIC 

$python cmic2.py path_to_output_directory_of_splitDNA2vec [options]

path_to_output_directory_of_splitDNA2vec is the directory generated by splitDNA2vec like  
k4_12_aug1000_v20_w10_e10_a0.025_mina0.0001_stride0. 

usage: cmic2.py [-h] [-o OUTPUT] [--seq_max_len SEQ_MAX_LEN] [-r RNN_TYPE] [--embedding_freeze EMBEDDING_FREEZE] [--shuffle_wv SHUFFLE_WV] [--debug_mode DEBUG_MODE] [--test_long TEST_LONG]
                data_dir

positional arguments:
  data_dir              This looks like exp_kmin_kmax/k4_11_aug1000_v20_w10_e10_a0.025_mina0.0001_splitDNA2vec

optional arguments:
  -h, --help            show this help message and exit

  -o OUTPUT, --output OUTPUT
                        specify the output file name (default: output.csv)

  --seq_max_len SEQ_MAX_LEN
                        The longer sequences are excluded. (default: 500)

  -r RNN_TYPE, --RNN_type RNN_TYPE
                        RNN, BiRNN, GRU, BiGRU, LSTM, BiLSTM (default: BiGRU)

  --embedding_freeze EMBEDDING_FREEZE
                        The embedding vectors are not updated. (default: True)

  --shuffle_wv SHUFFLE_WV
                        The word2vec mapping from k-mers to embedding vecotrs are shuffled before initializing the embedding layer. (default: False)

  --debug_mode DEBUG_MODE
                        Obsolete. (default: False)
                        
  --test_long TEST_LONG
                        Once trained models are obtained, this option is used to test longer CGIs. (default: False)



# Packages in environment:

| Name |                   Version |
| ---- | ---- |
| cudatoolkit   |            11.3.1 |             
gensim     |              4.1.2           
numpy        |             1.21.5          
pandas      |              1.4.2           
python      |              3.9.12         
pytorch      |             1.11.0        
scikit-learn |             1.0.2       
scipy       |              1.7.3      
tqdm        |              4.64.0    