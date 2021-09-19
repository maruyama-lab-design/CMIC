from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import numpy as np
import pandas as pd
import random
import math
from itertools import islice
import datetime
import time
import argparse

'''
    Generate word embedding vectors for k-mers. 
    We first divide all CGI sequences into 5 equal parts,
    then amplify them to N(2N, if use reverse complment) random-length k-mer sequences respectively.
    Then we use all random-length k-mer sequences of 5 parts to train word2vec to generate embedding vecrtors of k-mers.
    Two modes of make k-mer sequences:
    	0: gap = g overlapped k-mers(dna2vec)    
        1: non-overlapped k-mers(splitDNA2vec)

INPUT:  CGI sequences (M2M_for_RNN.fasta & M2U_for_RNN.fasta)
OUTPUT: cell_type_rc_N_mode_kmin_kmax (directory)            
		(i) random-length k-mer sequences and corresponding label[M2M(1) or M2U(0)]
		    (trv1data.csv & trv2data.csv & trv3data.csv & trv4data.csv & trv5data.csv)
		(ii) embedding vectors for k-mers (dna2vec.txt)            
		(iii) trained word2vec model (dna2vec.model)        
		...
'''

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

# The next is not called. But do not remove up to the completion. 

# def generate_random_kmer_seq_2(seq,k_min,k_max):
#     str = ''
#     i = 0
#     while i <= len(seq) - k_max + 1:
#         j  = random.randint(k_min,k_max)
#         kmer = seq[i:i+j]
#         str = str + kmer + ' '
#         i = i + j
#     return str

# merge two files into one
def mergefile(file1, file2,outwordfile):
    fo = open(outwordfile, 'w')
    for name in [file1, file2]:
        fi = open(name)
        while True:
            s = fi.read(16 * 1024)
            if not s:
                break
            fo.write(s)
        fi.close()
    fo.close()
    print('-----merged file1 and file2-----')

# train word2vec
def trainword2vec(Moutpath,Uoutpath,allpath,outpfile,embeddingfile,mincount,emd,w):
    mergefile(Moutpath,Uoutpath,allpath)
    # model = Word2Vec(LineSentence(allpath), size=emd, min_count=mincount,window=w,workers=4)
    model = Word2Vec(LineSentence(allpath), vector_size=emd, min_count=mincount,window=w,workers=4)
    model.save(outpfile)
    model = Word2Vec.load(outpfile)
    model.wv.save_word2vec_format(embeddingfile, binary=False)
    print('------dna2vec is trained---------')


def generate_data_for_RNN(Mdnafiletxt,Udnafiletxt):
    # Mcount = len(open(Mdnafiletxt, 'rU').readlines())
    Mcount = len(open(Mdnafiletxt, 'r').readlines())
    # Ucount = len(open(Udnafiletxt, 'rU').readlines())
    Ucount = len(open(Udnafiletxt, 'r').readlines())
    Mtemp = []
    Utemp = []
    # get data with M label
    t1 = open(Mdnafiletxt, 'r')
    lines = t1.readlines()
    for line in lines:
        Mtemp.append(line)
    t1.close()
    # get data with U label
    t2 = open(Udnafiletxt, 'r')
    lines = t2.readlines()
    for line in lines:
        Utemp.append(line)
    t2.close()
    # make  dataset
    alldata = pd.DataFrame(columns=('idx', 'CGI_seq', 'Label'))

    for i in range(Ucount):
        Label = 0
        CGI_seq = Utemp[i]
        alldata = alldata.append(pd.DataFrame({'idx': [i], 'CGI_seq': [CGI_seq],
                                           'Label': [Label]}), ignore_index=True)

    for j in range(Mcount):
        Label = 1
        CGI_seq = Mtemp[j]
        alldata = alldata.append(pd.DataFrame({'idx': [j], 'CGI_seq': [CGI_seq],
                                           'Label': [Label]}), ignore_index=True)
    # mess up
    alldata = alldata.sample(frac =1)
    return alldata

# mass up
def chunks(seqs, m):
    n = int(math.ceil(len(seqs) / float(m)))
    return [seqs[i:i + n] for i in range(0, len(seqs), n)]

# generate sequences of k-mer
def generate_sequence_of_kmer(method, seqs, path,outpath_all, outpath_f1, outpath_f2, outpath_f3, outpath_f4, outpath_test, k_min, k_max, s, times,rc,flag):
    dict = {'0': outpath_f1, '1': outpath_f2, '2': outpath_f3, '3':outpath_f4, '4':outpath_test}
    s = s
    times = times


    if method == 0:  
        # This case is the same as dna2vec. Namely, 
        # 1. The length of extracted k-mer is chosen at random. 
        # 2. A sliding window move to right with a given stride. 

        # generate_random_kmer_seq_0(DNA_seq_rc, k_min, k_max, s)

        # SlidingKmerFragmenter

        #shuffle
        random.shuffle(seqs)
        trv = chunks(seqs,5)
        for i in range(5):
            fseqs = trv[i]

            # Maybe, this file is just a record, not used in further codes. 
            file = open(path + '/' + flag + '_trv' + str(i)+'_data.txt', 'w')
            file.write(str(fseqs))
            file.close()

            trainseqs = []
            for j in range(len(fseqs)):
                DNA_seq = fseqs[j]
                # use Seq format
                temp_DNA = Seq(DNA_seq)
                DNA_seq_rc = temp_DNA.reverse_complement()
                for j in range(times):
                    split_result = generate_random_kmer_seq_0(DNA_seq, k_min, k_max, s)
                    tempstr = ''
                    for k in range(len(split_result)):
                        tempstr = tempstr + str(split_result[k]) + ' '
                    trainseqs.append(tempstr)
                if rc == True:
                    for q in range(times):
                        split_result = generate_random_kmer_seq_0(DNA_seq_rc, k_min, k_max, s)
                        tempstr = ''
                        for kk in range(len(split_result)):
                            tempstr = tempstr + str(split_result[kk]) + ' '
                        trainseqs.append(tempstr)
                outpath = dict[str(i)]
                fileObject1 = open(outpath, 'w')
                for ip in trainseqs:
                    p = str(ip)
                    fileObject1.write(p)
                    fileObject1.write('\n')
                fileObject1.close()
        # turn 5 files into 1 file
        mergefile(outpath_f1,outpath_f2,path+'/tempf1f2.txt')
        mergefile(path+'/tempf1f2.txt',outpath_f3,path+'/tempf1f2f3.txt')
        mergefile(path + '/tempf1f2f3.txt', outpath_f4, path + '/tempf1f2f3f4.txt')
        mergefile(path+'/tempf1f2f3f4.txt',outpath_test,outpath_all)

    elif method == 1:
        # splitDNA2vec. 

        # shuffle
        random.shuffle(seqs)
        trv = chunks(seqs, 5)
        for i in range(5):
            fseqs = trv[i]
            file = open(path + '/' + flag + '_trv' + str(i) + '_data.txt', 'w')
            file.write(str(fseqs))
            file.close()
            trainseqs = []
            for j in range(len(fseqs)):
                DNA_seq = fseqs[j]
                # use Seq format
                temp_DNA = Seq(DNA_seq)
                DNA_seq_rc = temp_DNA.reverse_complement()
                for k in range(times):
                    split_result = generate_random_kmer_seq_1(DNA_seq, k_min, k_max)
                    tempstr = ''
                    for kk in range(len(split_result)):
                        tempstr = tempstr + str(split_result[kk]) + ' '
                    trainseqs.append(tempstr)
                    #trainseqs.append(split_result)
                # r c
                if rc == True:
                    for j in range(times):
                        split_result = generate_random_kmer_seq_1(DNA_seq_rc, k_min, k_max)
                        tempstr = ''
                        for kk in range(len(split_result)):
                            tempstr = tempstr + str(split_result[kk]) + ' '
                        trainseqs.append(tempstr)
                    #trainseqs.append(split_result)

            outpath = dict[str(i)]
            fileObject1 = open(outpath, 'w')
            for ip in trainseqs:
                p = str(ip)
                fileObject1.write(p)
                fileObject1.write('\n')
            fileObject1.close()
        # turn 5 files into 1 file
        mergefile(outpath_f1, outpath_f2, path + '/tempf1f2.txt')
        mergefile(path + '/tempf1f2.txt', outpath_f3, path + '/tempf1f2f3.txt')
        mergefile(path + '/tempf1f2f3.txt', outpath_f4, path + '/tempf1f2f3f4.txt')
        mergefile(path + '/tempf1f2f3f4.txt', outpath_test, outpath_all)


    print('-----split dna sequence into sequence of k-mer-----')

# To ensure that the number of M and U is same
def matchnum(Mpath,Upath,path):
    Msequence = []
    Usequence = []
    for seq_record in SeqIO.parse(Mpath, "fasta"):
        tempseq = seq_record.seq
        seq = str(tempseq)
        Msequence.append(seq)
    for seq_record in SeqIO.parse(Upath, "fasta"):
        tempseq = seq_record.seq
        seq = str(tempseq)
        Usequence.append(seq)
    mcount = len(Msequence)
    ucount = len(Usequence)
    M = []
    U = []
    if mcount > ucount:
        temp = random.sample(range(1, mcount), ucount)
        for i in range(len(temp)):
            c = temp[i]
            M.append(Msequence[c])
        fileObject = open(path+'sampledM.txt', 'w')
        for ip in M:
            fileObject.write(ip)
            fileObject.write('\n')
        fileObject.close()
        return M,Usequence
    elif mcount < ucount :
        temp = random.sample(range(1, ucount), mcount)
        for i in range(len(temp)):
            c = temp[i]
            U.append(Usequence[c])
        fileObject = open(path + 'sampledU.txt', 'w')
        for ip in U:
            fileObject.write(ip)
            fileObject.write('\n')
        fileObject.close()
        return Msequence, U
    else:
        return Msequence,Usequence

# To readin M and U seuences
def read_seq_file(path):
    sequences = []
    for seq_record in SeqIO.parse(path, "fasta"):
        seq = str(seq_record.seq)
        sequences.append(seq)
    return sequences




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='splitDNA2vec hyperparameter defualt value.')
    # parser.add_argument('--title', help='filename of input', default='FGO2BMCGI')
    parser.add_argument('out_dir', help='outtpu directory name')
    parser.add_argument('--SEED', type=int, help='random seed for train', default=2020)
    parser.add_argument('--mincount', type=int, help='mincount option of word2vec', default=1)
    parser.add_argument('--emd', type=int, help='word embedding vector dimension', default=100)
    parser.add_argument('--w', type=int, help='neighbour window in word2vec', default=10)
    parser.add_argument('--g', type=int, help='gap size', default=1) # We don't use this
    parser.add_argument('--all_k_min', help='all optional values of k_min', default=[2,3,4])
    parser.add_argument('--all_k_max', help='all optional values of k_max', default=[4,5,6,7,8,9,10])
    parser.add_argument('--N', type=int, help='time of data augmentation', default=1)
    parser.add_argument('--mode', help='mode of generating kmer sequences', choices = [0,1], default=1)
    #type = 'Blastocyst_Maternal','Blastocyst_Paternal','FGoocyte','sperm'
    parser.add_argument('--cell_type', help='cell type', default='FGO2BM')
    parser.add_argument('--rc', help='use reverse complment or not', default=True)
    parser.add_argument('--d', type=int, help='minimum value of the difference', default=2)

    args = parser.parse_args()
    random.seed(args.SEED)
    # CGIpath: the path of input CGI sequences
    # CGIpath = '../data/' + args.title + '/'
    
    # CGIpath = './'
    CGIpath = args.out_dir


    for a in range(len(args.all_k_min)):
        for b in range(len(args.all_k_max)):
            k_min = args.all_k_min[a]
            k_max = args.all_k_max[b]
            con = k_max - k_min
            if con >= args.d:
                today = datetime.date.today()
                # if args.rc == True:
                #     kmerseqspath = CGIpath + str(today) + '/'+ args.cell_type +'_rc' + '_' + str(
                #         args.N) + '_' + str(args.mode)+ '_' + str(k_min) + '_' + str(k_max) + '/'
                # else:
                #     kmerseqspath = CGIpath + str(today) + '/'+ args.cell_type +'_' + str(
                #         args.N) + '_' + str(args.mode)+ '_' + str(k_min) + '_' + str(k_max) + '/'

                rev_comp = ''
                if args.rc == True:
                    rev_comp = '_rc'
                sub_dir = args.cell_type + rev_comp + '_' + str(args.N) + '_' + str(args.mode)+ '_' + str(k_min) + '_' + str(k_max)
                kmerseqspath = os.path.join(CGIpath, str(today), sub_dir)



                if not os.path.exists(kmerseqspath):
                    os.makedirs(kmerseqspath)

                # The next is made for runs of CMIC. 
                kmerseqspath_result = os.path.join(kmerseqspath, 'result')
                if not os.path.exists(kmerseqspath_result):
                    os.makedirs(kmerseqspath_result)

                Mpath = os.path.join(CGIpath, 'M2M_for_RNN.fasta')
                Upath = os.path.join(CGIpath, 'M2U_for_RNN.fasta')
                Moutpath = os.path.join(kmerseqspath, 'M2M_for_word2vec.txt')
                Uoutpath = os.path.join(kmerseqspath, 'M2U_for_word2vec.txt')
                # Moutpath_train = path + 'M2M_train.txt'
                Moutpath_f1 = os.path.join(kmerseqspath, 'M2M_trv_f1.txt')
                Moutpath_f2 = os.path.join(kmerseqspath, 'M2M_trv_f2.txt')
                Moutpath_f3 = os.path.join(kmerseqspath, 'M2M_trv_f3.txt')
                Moutpath_f4 = os.path.join(kmerseqspath, 'M2M_trv_f4.txt')
                Moutpath_f5 = os.path.join(kmerseqspath, 'M2M_trv_f5.txt')

                # Uoutpath_train = path + 'M2U_train.txt'

                Uoutpath_f1 = os.path.join(kmerseqspath, 'M2U_trv_f1.txt')
                Uoutpath_f2 = os.path.join(kmerseqspath, 'M2U_trv_f2.txt')
                Uoutpath_f3 = os.path.join(kmerseqspath, 'M2U_trv_f3.txt')
                Uoutpath_f4 = os.path.join(kmerseqspath, 'M2U_trv_f4.txt')
                Uoutpath_f5 = os.path.join(kmerseqspath, 'M2U_trv_f5.txt')

                # If we want the same number of M and U sequences
                # Mseq, Useq = matchnum(Mpath, Upath, path)

                # Mseq, Useq = read_seq_file(Mpath, Upath, kmerseqspath)
                Mseq = read_seq_file(Mpath)
                Useq = read_seq_file(Upath)

                # split dna sequence into sequence o d k-mer
                print(str(k_min), '~', str(k_max), ':-------start split------------')

                generate_sequence_of_kmer(args.mode, Mseq, kmerseqspath, Moutpath, Moutpath_f1, Moutpath_f2,
                                          Moutpath_f3,
                                          Moutpath_f4, Moutpath_f5, k_min, k_max, args.g, args.N, args.rc,
                                          flag='M')
                generate_sequence_of_kmer(args.mode, Useq, kmerseqspath, Uoutpath, Uoutpath_f1, Uoutpath_f2,
                                          Uoutpath_f3,
                                          Uoutpath_f4, Uoutpath_f5, k_min, k_max, args.g, args.N, args.rc,
                                          flag='U')

                # train dna2vec model（word2vec）
                print(str(k_min), '~', str(k_max), ':-------start train word2vec------------')
                allpath = os.path.join(kmerseqspath, 'newall.txt')
                outpfile = os.path.join(kmerseqspath, 'dna2vec.model')
                embeddingfile = os.path.join(kmerseqspath, 'dna2vec.txt')
                trainword2vec(Moutpath, Uoutpath, allpath, outpfile, embeddingfile, args.mincount, args.emd, args.w)

                # make data for RNN

                trv1data = generate_data_for_RNN(Moutpath_f1, Uoutpath_f1)
                trv2data = generate_data_for_RNN(Moutpath_f2, Uoutpath_f2)
                trv3data = generate_data_for_RNN(Moutpath_f3, Uoutpath_f3)
                trv4data = generate_data_for_RNN(Moutpath_f4, Uoutpath_f4)
                trv5data = generate_data_for_RNN(Moutpath_f5, Uoutpath_f5)

                trv1datapath = os.path.join(kmerseqspath, 'trv1data.csv')
                trv2datapath = os.path.join(kmerseqspath, 'trv2data.csv')
                trv3datapath = os.path.join(kmerseqspath, 'trv3data.csv')
                trv4datapath = os.path.join(kmerseqspath, 'trv4data.csv')
                trv5datapath = os.path.join(kmerseqspath, 'trv5data.csv')

                trv1data.to_csv(trv1datapath, columns=['idx', 'CGI_seq', 'Label'], index=0)
                trv2data.to_csv(trv2datapath, columns=['idx', 'CGI_seq', 'Label'], index=0)
                trv3data.to_csv(trv3datapath, columns=['idx', 'CGI_seq', 'Label'], index=0)
                trv4data.to_csv(trv4datapath, columns=['idx', 'CGI_seq', 'Label'], index=0)
                trv5data.to_csv(trv5datapath, columns=['idx', 'CGI_seq', 'Label'], index=0)

                datasets = [trv1data, trv2data, trv3data, trv4data, trv5data]
                # datasets = [df1, df2, df3, df4]
                allcgikmercsvpath = os.path.join(kmerseqspath, 'allcgikmer.csv')
                tempt = pd.DataFrame(columns=['idx', 'CGI_seq', 'Label'])
                for i, v in enumerate(datasets):
                    tempt = tempt.append(v)
                tempt.to_csv(allcgikmercsvpath, columns=['idx', 'CGI_seq', 'Label'], index=0)
                print(str(k_min), '~', str(k_max), ':-------csv ok------------')