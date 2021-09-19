1. Suppose that 
m.fasta and u.fasta is your original FASTA files of methylated and unmethylated 
sequences.  

2. Extract short sequences from m.fasta and u.fasta by 
Library/mSeqGenerator/Library/makeLimitedLengthCGI.py:
The output filenames are 
M2M_for_RNN.fasta and 
M2U_for_RNN.fasta, 
stored in a specified output directory. 

$ python makeLimitedLengthCGI.py data-FGO2BM/m.fasta  data-FGO2BM/u.fasta data-FGO2BM --seq_len_upper_bound 500

3. The next is how to use splitDNA2vec. 
Option --title specifies the directory name at ../data. 
In the example, it is data-FGO2BM. 

$ Library/splitDNA2vec.py --title data-FGO2BM

Multiple directories are generated according to used parameters: 

The format of such directories: [cell_type]_rc_[N_mode]_[mode]_[kmin]_[kmax] (directory)            
where mode is ... 
like FGO2PB_rc_1_1_2_4

(i) 
trv1data.csv 
trv2data.csv 
trv3data.csv 
trv4data.csv 
trv5data.csv
: random-length k-mer sequences and corresponding label[M2M(1) or M2U(0)]

(ii) 
dna2vec.txt
: embedding vectors for k-mers 

(iii) 
dna2vec.model
: trained word2vec model 

(iv)
allcgikmer.csv 
: ?

The files of (i)-(iv) are called in the model-learning program. 

(v)
There are many other files...


4. 
MethGRU-model.py (MethGRU is the previous toolname. It is currently CMIC.) requires 
the following input files: 

trv1data.csv
trv2data.csv
trv3data.csv
trv4data.csv
trv5data.csv
and
allcgikmer.csv