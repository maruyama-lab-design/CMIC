from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import argparse
import os

'''
Extract all CGIs whose lengths are less than 500 bp.

INPUT: m.fasta and u.fasta. 
OUTPUT: M2M_for_RNN.fasta & M2U_for_RNN.fasta
'''

def makedata(seq_len_upper_bound, in_filename, out_filename):
    short_seqs = []
    length  = []
    for seq_record in SeqIO.parse(in_filename, "fasta"):
        length.append(len(seq_record))
        if len(seq_record) < seq_len_upper_bound:
            rec = SeqRecord(seq=seq_record.seq, id=seq_record.id, description=seq_record.description)
            short_seqs.append(rec)
    SeqIO.write(short_seqs, out_filename, "fasta")
    return length

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose CGI sequences to use')
    parser.add_argument('m_fasta', help='name of methylated FASTA file')
    parser.add_argument('u_fasta', help='name of unmethylated FASTA input')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--seq_len_upper_bound', help='length upper bound of CGIs', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    makedata(args.seq_len_upper_bound, args.m_fasta, os.path.join(args.out_dir, "M2M_for_RNN.fasta"))
    makedata(args.seq_len_upper_bound, args.u_fasta, os.path.join(args.out_dir, "M2U_for_RNN.fasta"))




