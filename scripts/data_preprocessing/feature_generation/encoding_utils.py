import numpy as np
import pickle

with open('scripts/data_preprocessing/feature_generation/bendability.pkl','rb') as f:
    BEND_DICT = pickle.load(f)

with open('scripts/data_preprocessing/feature_generation/propellerTwist.pkl','rb') as f:
    PROP_DICT = pickle.load(f)

def LabelEncoding(seqs):
    # Make seqs a numpy array of size (n_seqs, n_nts)
    seq_arr = np.array(seqs, dtype=np.unicode_)
    seq_arr = seq_arr.view('U1').reshape((seq_arr.size, -1))
    print("seq_arr shape:",seq_arr.shape)

    # Get indicies of unique values
    nts, idx = np.unique(seq_arr,return_inverse=True)
    print("Unique nucleotides in sample:", ', '.join(nts.tolist()))
    idx = idx.reshape(seq_arr.shape)

    return idx


def OneHotEncoding(label_enc):
    # Identity matrix use as rows
    e = np.eye(4)

    # Perform one hot encoding size (n_seqs, 4, n_nts)
    return e[label_enc].transpose(0,2,1)


def EIIP(label_enc):
    eiip_mapping = np.array([0.1260, 0.1340,  0.0806, 0.1335])
    rows = np.zeros((label_enc.shape),dtype=int)
    return eiip_mapping[None,:][rows,label_enc][:,None,:]  # (n_seqs, 1, n_nts)


def ZCurve(label_enc):
    pur_pyr = np.array([[1,-1,1,-1]]) # A=1, C=-1, G=1, T=-1
    amino_keto = np.array([[1,1,-1,-1]]) # A=1, C=1, G=-1, T=-1
    weak_strong = np.array([[1,-1,-1,1]]) # A=1, C=-1, G=-1, T=1
    zcurve_enc = np.vstack((pur_pyr,amino_keto,weak_strong))

    zcurve_shape = (label_enc.shape[0],3,label_enc.shape[1])
    rows = np.zeros((zcurve_shape),dtype=int)
    
    # Indicies to make the mapping
    z_axis_idx1 = rows + np.arange(3)[None,:,None]
    z_axis_idx2 = rows + label_enc[:,None,:]

    # Perform mapping
    noncum_zcurve_mapping = zcurve_enc[None,:,:][rows,z_axis_idx1,z_axis_idx2] # # (n_seqs, 3, n_nts)

    return np.cumsum(noncum_zcurve_mapping,axis=2)

def Bendability(seqs, size=3):
    seq_bends = np.zeros((len(seqs),len(seqs[0])))
    for row_idx, seq in enumerate(seqs):
        # out = [0, 0] # , 0]
        for x in range(0, len(seq) - size):
            kmer = seq[x:x + size]
            seq_bends[row_idx,x+2] = BEND_DICT[kmer]
    return seq_bends[:,None,:]

def PropellerTwist(seqs, size=2):
    seq_props = np.ones((len(seqs),len(seqs[0])))
    seq_props[:,:1] *= -12.6
    # out = ['-12.6', '-12.6']
    for row_idx, seq in enumerate(seqs):
        for x in range(0, len(seq) - size):
            kmer = seq[x:x + size]
            seq_props[row_idx,x+1] = PROP_DICT[kmer]
    return seq_props[:,None,:]

