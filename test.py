'''
Test a trained MethGRU model using a new dataset
There are two parts in this file
(1) Generate test data
Generate one k-mer sequence for one DNA sequence
and then save k-mer sequences in txt file
(2) Test model

INPUT: a model file(.pt) & DNA sequences(.fasta)
OUTPUT:  prediction accuracy

'''
import torch
from torchtext import data
from torchtext.data import Field
from torchtext.vocab import Vectors
from torchtext.data import TabularDataset
from train0819 import classifier
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from itertools import islice
import random
import argparse
import os
import argparse
import numpy as np
import pandas as pd

# split dna seqences fasta into many files
def splitfile(Readfasta,outputpath):
    i = 0
    for seq_record in SeqIO.parse(Readfasta, "fasta"):
        rec = SeqRecord(seq_record.seq, seq_record.id, seq_record.description)
        output = outputpath+str(i)+'.fasta'

        SeqIO.write(rec, output, "fasta")
        i = i + 1
    return i

# Select k-mer sequences whose length is less than 500bp
def checklength(Readfasta,outputpath):
    recs = []
    for seq_record in SeqIO.parse(Readfasta, "fasta"):
        rec = SeqRecord(seq_record.seq,seq_record.id,seq_record.description)
        recs.append(rec)
    SeqIO.write(recs, outputpath, "fasta")



# split long sequenes into k-mers
#function for split dna sequence
#function for split dna sequence mode:0
def generate_random_kmer_seq_0(seq,k_min,k_max,s):
    return [str(seq[i: i + random.randint(k_min, k_max)]) for i in range(0,(len(seq) - k_max + 1),s)]

#function for split dna sequence mode:1
def generate_random_kmer_seq_1(seq,k_min,k_max):
    #seq = seq[random.randint(0,k_min):]  # randomly offset the beginning to create more variations
    return list(random_chunks(seq, k_min, k_max))

def random_chunks(seq, min_chunk, max_chunk):
    it = iter(seq)
    while True:
        head_it = islice(it, random.randint(min_chunk, max_chunk))
        nxt = '' . join(head_it)
        if len(nxt) >= min_chunk:
            yield nxt
        else:
            break

# generate sequences of k-mers
# one k-mer sequence for one DNA sequence
def generate_sequence_of_kmer(fastafile,k_low,k_high,times,rc,mode):
    kmerseqs = []
    if mode == 0:
        for seq_record in SeqIO.parse(fastafile, "fasta"):
            tempseq = seq_record.seq
            seq = str(tempseq)
            temp_DNA = Seq(seq)
            seq_rc = temp_DNA.reverse_complement()
            for i in range(times):
                split_result = generate_random_kmer_seq_0(seq, k_low, k_high)
                kmerseqs.append(split_result)
                if rc == True:
                    split_result = generate_random_kmer_seq_0(seq_rc, k_low, k_high)
                    kmerseqs.append(split_result)
        return kmerseqs

    elif mode ==1:
        for seq_record in SeqIO.parse(fastafile, "fasta"):
            tempseq = seq_record.seq
            seq = str(tempseq)
            temp_DNA = Seq(seq)
            seq_rc = temp_DNA.reverse_complement()
            for i in range(times):
                split_result = generate_random_kmer_seq_1(seq, k_low, k_high)
                kmerseqs.append(split_result)
                if rc == True:
                    split_result = generate_random_kmer_seq_1(seq_rc, k_low, k_high)
                    kmerseqs.append(split_result)
        return kmerseqs


def generate(num,outputfilepath,args):
    for i in range(num):
        fastafile = M2Mfiles+str(i)+'.fasta'
        kmerseqs = generate_sequence_of_kmer(fastafile, k_low=args.k_min, k_high=args.k_max,
                                             times=args.N,rc = args.rc, mode = args.mode)
        f = open(outputfilepath+str(i)+'.txt', 'w')
        for x in kmerseqs:
            f.write(str(x)+ "\n")
        f.close()


def predict(model, sentence):
    indexed = []
    for i in range(len(sentence)):
        indexed.append(TEXT.vocab.stoi[sentence[i]])
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to('cpu')              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction
    return prediction.item()



def maketestseqs(kmerfilepath):
    kmerseqs = []  # seqs for test
    for line in open(kmerfilepath):
        l = len(line)
        # kmers = line.replace(',',' ')
        patt = re.compile(r"[^a-zA-Z- ]+")
        ll = patt.sub('', line)
        kmerseqs.append(ll)
    return kmerseqs

def testresult(mtxtfile,utxtfile):
    m = maketestseqs(mtxtfile)
    u = maketestseqs(utxtfile)
    mre = []
    mre = [1 in range(len(m)) ]
    for i in range(len(m)):
        seq = m[i]
        sentence = (seq.split(' '))
        pre_res = predict(model, sentence)
        p = round(pre_res)
        mre.append(p)
    ure = []
    for i in range(len(u)):
        seq = u[i]
        sentence = (seq.split(' '))
        pre_res = predict(model, sentence)
        p = round(pre_res)
        ure.append(p)
def startpredict(filepath):
    num = len([lists for lists in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, lists))])
    re = []
    for i in range(num-1):
        kmerfilepath = filepath + str(i) + '.txt'
        kmerseqs = maketestseqs(kmerfilepath)
        ure = []
        for i in range(len(kmerseqs)):
            seq = kmerseqs[i]
            sentence = (seq.split(' '))
            pre_res = predict(model, sentence)
            p = round(pre_res)
            ure.append(p)
        #f = ure.count(0)
        #t = ure.count(1)
        u = np.array(ure)
        r = np.mean(u)
        rr = np.round(r)
        re.append(rr)
    return re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test trained MethGRU model')
    parser.add_argument('--title', help='filename of input', default='FGO2BMCGI')
    parser.add_argument('--SEED', type=int, help='random seed for train', default=2020)
    parser.add_argument('--g', type=int, help='gap size', default=1)  # We don't use this
    parser.add_argument('--k_min', help='the value of k_min', default=3)
    parser.add_argument('--k_max', help='the value of k_max', default=9)
    parser.add_argument('--N', type=int, help='time of data augmentation', default=10)
    parser.add_argument('--mode', help='mode of generating kmer sequences', choices=[0, 1], default=1)
    parser.add_argument('--H', type=int, help='hidden state dimension of RNN', default=256)
    parser.add_argument('--dp', type=float, help='dropout rate', default=0.5)
    parser.add_argument('--D', type=int, help='word embedding vector dimension', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=10e-5)
    parser.add_argument('--rc', help='use reverse complment or not', default=True)
    parser.add_argument('--title', help='filename of input', default='FGO2BMCGI')
    parser.add_argument('--time_generateRLKS', help='filename of input', default='2021-03-16')

    args = parser.parse_args()

    random.seed(args.SEED)
    # CGIpath: the path of input CGI sequences
    # CGIpath = '../data/' + args.title + '/'

    # readin M2M and M2U DNA sequences
    rootpath = 'C:/Users/LI/Desktop/methylationdata/FGO2BM_cheklong/'
    mseqspath = rootpath + 'M2Mover500.fasta'
    useqspath = rootpath + 'M2Uover500.fasta'

    M2Mfiles = rootpath + 'M2M/fasta/'
    M2Ufiles = rootpath + 'M2U/fasta/'
    outputM2Mkmerseq = rootpath + 'M2M/kmerseq/'
    outputM2Ukmerseq = rootpath + 'M2U/kmerseq/'

    if not os.path.exists(M2Mfiles):
        os.makedirs(M2Mfiles)
    if not os.path.exists(M2Ufiles):
        os.makedirs(M2Ufiles)
    if not os.path.exists(outputM2Mkmerseq):
        os.makedirs(outputM2Mkmerseq)
    if not os.path.exists(outputM2Ukmerseq):
        os.makedirs(outputM2Ukmerseq)
    # One fasta file contains one seq
    Mnum = splitfile(mseqspath, M2Mfiles)
    Unum = splitfile(useqspath, M2Ufiles)
    generate(Mnum, outputM2Mkmerseq, args)
    generate(Unum, outputM2Ukmerseq, args)


    # load weights
    csvpath = 'C:/Users/LI/Desktop/methylationdata/New/2020-09-27/FGO2BM_rc_1000_3_9'
    path = csvpath + '/result/256_10_0.5_0.0001_jiumingadna2vec_modelwhole.pt'
    nm = 'dna2vec.txt'

    c = csvpath + '/'
    allcgikmercsv = csvpath+'/allcgikmer.csv'

    # ------------Configuration---------
    tokenize = lambda x: x.split()
    TEXT = Field(tokenize=tokenize, include_lengths=True, batch_first=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)
    datafields = [("idx", None),  # we won't be needing the idx, so we pass in None as the field
                  ("CGI_seq", TEXT),
                  ("Label", LABEL)]
    allcgi = TabularDataset(
        path=allcgikmercsv,  # the root directory where the data lies
        format='csv',
        skip_header=True,
        # if csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data
        fields=datafields)

    vectors = Vectors(name=nm, cache=c)
    TEXT.build_vocab(allcgi, vectors=vectors)
    # LABEL.build_vocab(trn)
    size_of_vocab = len(TEXT.vocab)

    # Setting of model
    model = classifier(size_of_vocab, embedding_dim=args.D, hidden_dim=args.H, output_dim=1, n_layers=1,
                       bidirectional=True, dropout=args.dp)
    model.load_state_dict(torch.load(path))
    model.eval()
    device = 'gpu'

    mre = startpredict(outputM2Mkmerseq)
    ure = startpredict(outputM2Ukmerseq)
    TP = mre.count(1)
    FN = mre.count(0)
    TN = ure.count(0)
    FP = ure.count(1)
    acc = (TP+TN)/(TP+TN+FP+FN)
    # precision = TP / ((TP + FP))
    # recall = TP / ((TP + FN) )
    # FPR = FP / ((TP + TN))
    # F = (2 * precision * recall) / ((precision + recall))
    print('Test accuracy is = ',acc)





