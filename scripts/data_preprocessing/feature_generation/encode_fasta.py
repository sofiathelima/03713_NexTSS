from encoding_utils import *

import os
import sys
import pickle
import argparse

import numpy as np
from sklearn.model_selection import train_test_split


def LabelEncodingFromFasta(fasta_path):
    """
    Agrs:
    - fasta_path: path to fast file

    Return:
    - label_encs (np.array: (n_seqs, n_nts)): A label encoding of the sequences from the fasta files. 
        ex. ACGT -> 0123
    - names (list): A list of names for each sequence
    """

    # Save lines of file
    with open(fasta_path, 'r') as f:
        data = f.readlines()
    
    # extract all sequences i.e all odd lines of file (1-based)
    names = [x.strip() for x in data[::2]]
    # print('first name:',names[0])

    # extract all sequences i.e all even lines of file (1-based)
    seqs = [x.strip().upper() for x in data[1::2]]

    # Get label encoding for all samples
    seq_label_enc = LabelEncoding(seqs)

    # print('Shape seq labels:',seq_label_enc)
    # print('Length of names:',len(names))

    return seq_label_enc, seqs


def PhyloPArray(sample_phylo):
    # Read data
    with open(sample_phylo, 'r') as f:
        data = f.readlines()

    # Extract nanmes and scores
    headers = data[::2]
    scores = [x.strip().split(' ') for x in data[1::2]]

    tot = 0
    for x in scores:
        tot += len(x)

    print('average len sequence:',tot/len(scores))
    # scores = np.array([x.strip().split(' ') for x in data[1::2]], dtype=float)

    scores = scores[:,None,:]

    return scores
    


def GenerateV1Features(seq_label_encs):
    """
    Comput the features for regions from label encodings
    Args:
    - label_encs (np.array: (n_seqs, n_nts)): A label encoding of the sequences from the fasta files.
    Return:
    - feats (np.array: (n_seqs, n_feats, n_nts))
    """

    oh_enc = OneHotEncoding(seq_label_encs) # (n_seqs, 4, n_nts)
    eiip_enc = EIIP(seq_label_encs) # (n_seqs, 1 n_nts)
    zcurve_enc = ZCurve(seq_label_encs) # (n_seqs, 3, n_nts)

    print('One hot encoding shape:',oh_enc.shape)
    print('EIIP',eiip_enc.shape)
    print('Z Curve',zcurve_enc.shape)

    final_enc = np.concatenate([oh_enc,eiip_enc,zcurve_enc],axis=1)
    print("final encoding shape:",final_enc.shape)

    return final_enc #


def GenerateV2Features(seq_label_encs, seqs):
    """
    Comput the features for regions from label encodings
    Args:
    - label_encs (np.array: (n_seqs, n_nts)): A label encoding of the sequences from the fasta files.
    Return:
    - feats (np.array: (n_seqs, n_feats, n_nts))
    """

    oh_enc = OneHotEncoding(seq_label_encs) # (n_seqs, 4, n_nts)
    eiip_enc = EIIP(seq_label_encs) # (n_seqs, 1 n_nts)
    zcurve_enc = ZCurve(seq_label_encs) # (n_seqs, 3, n_nts)
    bends_enc = Bendability(seqs)
    props_enc = PropellerTwist(seqs)

    print('One hot encoding shape:',oh_enc.shape)
    print('EIIP:',eiip_enc.shape)
    print('Z Curve:',zcurve_enc.shape)
    print('Bendability:',bends_enc.shape)
    print('PropellerTwist:',props_enc.shape)
    # print('PhyloP:',zcurve_enc.shape)

    final_enc = np.concatenate([oh_enc,eiip_enc,zcurve_enc,bends_enc,props_enc],axis=1)
    print("final encoding shape:",final_enc.shape)

    return final_enc #


def SaveFeaturesToFile(feats,save_dir,label,feat_desc=None):
    # Make directory structure
    split_dirs = ['train','test','valid']
    split_dir_paths = []
    for dir in split_dirs:
        split_dir_path = save_dir+f'/{dir}/Class_{label}'
        if not os.path.exists(split_dir_path):
            os.makedirs(split_dir_path)
        split_dir_paths.append(split_dir_path)

    # split data into train test validation sets (70/20/10)
    train_X, test_X = train_test_split(feats,test_size=0.3) # 70/30 train test
    test_X, val_X = train_test_split(test_X,test_size=0.33) # 20/10 test,val

    data_splits = [train_X, test_X, val_X]

    # Save data to file directories
    for data_dir, data_split in zip(split_dir_paths,data_splits):
        samples_dict = {
                "data":data_split,
                "label":label,
            }
        with open(data_dir+f'/samples_dict.pkl','wb') as f:
                pickle.dump(samples_dict,f)



if __name__ == "__main__":

    sample_fasta = sys.argv[1]
    save_dir = sys.argv[2]
    label = sys.argv[3]

    seq_label_encs, seqs = LabelEncodingFromFasta(sample_fasta)

    # final_feats = GenerateV1Features(seq_label_encs)
    final_feats = GenerateV2Features(seq_label_encs,seqs)

    SaveFeaturesToFile(final_feats,save_dir,label,feat_desc=None)

    