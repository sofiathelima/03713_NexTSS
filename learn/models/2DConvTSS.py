import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import random
import numpy as np
import argparse
import sys
import time

import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns 

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class DummyDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
 
    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target
    

class ModelDataset(Dataset):
    def __init__(self, species_dir,split='train'):
        # convert into PyTorch tensors and remember them
        data, labels = [], []
        split_dir = species_dir+f'/{split}'

        for class_dir in os.listdir(split_dir):
            with open(split_dir+f'/{class_dir}/samples_dict.pkl','rb') as f:
                data_dict = pickle.load(f)
                class_label = int(data_dict['label'])

                data.append(data_dict['data'])
                labels.append(class_label * np.ones(len(data_dict["data"])))
        
        self.data = np.concatenate(data,axis=0)
        self.labels = np.concatenate(labels)

        shuffle_idx = np.arange(len(self.labels))
        np.random.shuffle(shuffle_idx)

        self.data = self.data[shuffle_idx]
        self.data = self.data[:,None,:,:]
        self.labels = self.labels[shuffle_idx]
        print('Data shape:',self.data.shape)
        print('Labels shape:',self.labels.shape)


    def __len__(self):
        # this should return the size of the dataset
        return len(self.data)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.data[idx]
        target = self.labels[idx]
        return features, target



class NexTSS(nn.Module):
    def __init__(self,feat_num):
        super(NexTSS, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, (feat_num,16))
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d((1,2))
        self.drop1 = nn.Dropout(0.2)
        #Dropout
        self.conv2 = nn.Conv2d(20, 10, (1,12)) 
        self.bn2 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d((1,2))
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(1400, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(120, 60)
        self.bn4 = nn.BatchNorm1d(60)
        self.drop4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(60, 1)

    def forward(self, x):
        
        # print('before Conv1',x.shape)
        x = F.leaky_relu(self.conv1(x))
        # print('before Batch1',x.shape)
        x = self.bn1(x)
        # print('before Pool1',x.shape)
        x = self.pool1(x)
        x = self.drop1(x)
        # print('before Conv2',x.shape)
        x = F.leaky_relu(self.conv2(x))
        # print('before Batch2',x.shape)
        x = self.bn2(x)
        # print('before Pool2',x.shape)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # print('before Flatten',x.shape)
        x = x.view(x.size()[0], -1) # flatten layer
        # print('before Linear1',x.shape)
        x = F.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        x = self.drop3(x)
        # print('before Linear2',x.shape)
        x = F.leaky_relu(self.fc4(x))
        x = self.bn4(x)
        x = self.drop4(x)
        # print('before Linear3',x.shape)
        x = self.fc5(x)


        # 60 training (3 mini-batches)
        # ########## For first two mini-batches:
        # before Conv1 torch.Size([16, 1, 4, 600])
        # before Batch1 torch.Size([16, 20, 1, 585])
        # before Pool1 torch.Size([16, 20, 1, 585])
        # before Conv2 torch.Size([16, 20, 1, 292])
        # before Batch2 torch.Size([16, 10, 1, 281])
        # before Pool2 torch.Size([16, 10, 1, 281])
        # before Flatten torch.Size([16, 10, 1, 140])
        # before Linear1 torch.Size([16, 1400])
        # before Linear2 torch.Size([16, 120])
        # before Linear3 torch.Size([16, 60])
        
        # ########## For last mini-batch:
        # before Conv1 torch.Size([8, 1, 4, 600])
        # before Batch1 torch.Size([8, 20, 1, 585])
        # before Pool1 torch.Size([8, 20, 1, 585])
        # before Conv2 torch.Size([8, 20, 1, 292])
        # before Batch2 torch.Size([8, 10, 1, 281])
        # before Pool2 torch.Size([8, 10, 1, 281])
        # before Flatten torch.Size([8, 10, 1, 140])
        # before Linear1 torch.Size([8, 1400])
        # before Linear2 torch.Size([8, 120])
        # before Linear3 torch.Size([8, 60])
        
        x = torch.sigmoid(x)
        
        return x
    

def load_split(split_path, b):
    
    class_paths = os.listdir(split_path)
    print('len(class_paths)' ,len(class_paths), class_paths)
    
    fnames_class1 = os.listdir(os.path.join(split_path, class_paths[0]))
    num_samples = len(fnames_class1)
    f_path = os.path.join(split_path,class_paths[0],fnames_class1[0])
    f = open(f_path,'rb')
    data = pickle.load(f)
    f.close()
    num_features = data.shape[0]
    window_size = data.shape[1]
    
    feature_data = np.zeros((len(class_paths), num_samples, num_features, window_size))
    label_data = np.zeros((len(class_paths), num_samples,1))
    
    for gt, label in enumerate(class_paths):
        print(label)
        class_dir = os.path.join(split_path, label)
        fnames = os.listdir(class_dir)
        print('len(fnames)' ,len(fnames))
        
        f_path = os.path.join(class_dir,fnames[0])
        f = open(f_path,'rb')
        data = pickle.load(f)
        f.close()
        num_features = data.shape[0]
        window_size = data.shape[1]
        
        label_data[gt,:,0] = np.full(len(fnames), gt)
    
        for i, fname in enumerate(fnames):
            f_path = os.path.join(class_dir,fname)
            f = open(f_path,'rb')
            data = pickle.load(f)
            f.close()
            
            feature_data[gt,i,:,:] = data
            
    X_data = np.concatenate((feature_data[0,:,:,:],feature_data[1,:,:,:]))
    X_data = np.expand_dims(X_data, 1)
    #testing only 1 sample
    # X_data = X_data[0:2,:,:,:]
    print('X_data.shape',X_data.shape)
    y_data = np.concatenate((label_data[0],label_data[1]))
    #testing only 1 sample
    # y_data = y_data[0:2]
    print('y_data.shape',y_data.shape)
    
    dataset = DummyDataset(X_data, y_data)
    loader = DataLoader(dataset, shuffle=True, batch_size=b)
    
    return loader

def load_dataset(species_dir, b):
    
    iters = []
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(species_dir, split)
        
        loader = load_split(split_path, b)
        iters.append(loader)
        
    
    return iters

def load_model_split(species_dir,split,bs):
    dataset = ModelDataset(species_dir,split)
    return DataLoader(dataset, shuffle=True, batch_size=bs)


def load_model_datasets(species_dir,bs):
    iters = []
    
    for split in ['train', 'valid', 'test']:
        iters.append(load_model_split(species_dir,split,bs))
    
    return iters

def calculate_accuracy(fx, y):
    preds = fx > 0.5
    # correct = preds.eq(y.view_as(preds)).sum()
    # acc = correct.float()/preds.shape[0]
    acc = accuracy_score(y, preds)
    return acc

def train(model, device, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.float().to(device)
        y = y.float().to(device)
        
        optimizer.zero_grad()
                
        fx = model(x).squeeze()
        
        loss = criterion(fx, y)
        
        acc = calculate_accuracy(fx, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)   

def evaluate(model, device, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()

    pred_batches = []
    
    with torch.no_grad():
        for (x, y) in iterator:

            x = x.float().to(device)
            y = y.float().to(device)

            fx = model(x).squeeze()
            pred_batches.append(fx)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    preds = torch.cat([y for y in pred_batches])

    # print('type pred:',type(preds[0]))
    # print("# positive samples:", (preds > 0.5).sum())
    # print('preds:',preds[:20])
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), np.array(preds)


def visualize(X, t, v, t_loss, v_loss, fname):

    # Training accuracy
    fig = plt.figure()
    plt.title("Training Deep Learning model")
    plt.plot(X,t, label='training data')
    plt.plot(X,v, label='validation data')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(fname)

    # Training loss
    fig = plt.figure()
    plt.title("Training Deep Learning model")
    plt.plot(X,t_loss, label='training data')
    plt.plot(X,v_loss, label='validation data')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(fname+'_loss')

    return

def visualize_testing(y_true, y_pred, fname):
    print(y_true.shape)
    print(y_pred.shape)

    # y_true = np.squeeze(y_true)
    # y_pred = np.squeeze(y_pred)
    # print(y_true.shape)
    # print(y_pred.shape)

    # print('y_true:',*y_true[:10])
    # print('y_pred:',*y_pred[:10])
    print("# positive samples:", (y_pred > 0.5).sum())

    # print(type(y_true[0]))
    # print(type(y_pred[0]))

    # ROC Curve
    fig = plt.figure()
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="ROC curve (AUC = {0:0.3f})".format(auc_score))
    plt.plot(fpr, fpr, '--')
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.legend()
    plt.savefig(fname+'_roc')

    y_pred = (y_pred > 0.5).astype(int)

    # Confusion Matrix
    fig = plt.figure()
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(f'cf_matrix = {cf_matrix}')
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    counts = [f'{value}/{str(np.sum(cf_matrix))}' for value in cf_matrix.flatten()]

    prcts = [f'{round(val,2)*100}%' for val in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,prcts)]

    if len(labels) > 1:
      labels = np.asarray(labels).reshape(2,2)
      sns.heatmap(cf_matrix, annot=labels, fmt='')
    else:
        sns.heatmap(cf_matrix, fmt='')
    plt.savefig(fname+'_cf')


    # Binary Classif Metrics
    fig = plt.figure()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    y_plot = [acc, f1, prec, rec]
    x_plot = ['Accuracy','F1 Score','Precision', 'Recall']
    plt.bar(x_plot, y_plot)
    # plt.plot(fpr, tpr, label="ROC curve (AUC = {0:0.3f})".format(auc_score))
    # plt.plot(fpr, fpr, '--')
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(fname+'_metrics')

    return

# organism_dir = '/ocean/projects/bio230007p/psimpson/group4/data/data_v1/human'
organism_dir = sys.argv[1]
mouse_dir = sys.argv[2]
feat_num = int(sys.argv[6])

b_size = 512
[train_iterator, test_iterator, valid_iterator] = load_model_datasets(organism_dir, b_size)

# mouse_dir = '/ocean/projects/bio230007p/psimpson/group4/data/data_v1/mouse'
b_size = 256
[mouse_train_iterator, mouse_test_iterator, mouse_valid_iterator] = load_model_datasets(mouse_dir, b_size)

# train_features, train_labels = next(iter(train_iterator))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

model = NexTSS(feat_num)

model.to(device)

optimizer = optim.Adam(model.parameters())

criterion = nn.BCELoss()

load_checkpoint, skip_train = False, False

#You are recommended to store check points
# if len(sys.argv) > 1:
#     if sys.argv[1] == "c":
#         load_checkpoint = True
#     elif sys.argv[1] == "s":
#         skip_train = True  #Proceed directly to testing


### Training ###
MODEL_SAVE_DIR = sys.argv[3]
GRAPH_SAVE_DIR = sys.argv[4]
EPOCHS = int(sys.argv[5])

# SAVE_DIR = '/ocean/projects/bio230007p/psimpson/group4/learn/models'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'net.pt')

best_valid_loss = float('inf')
time_curr = time.time()

if not os.path.isdir(f'{MODEL_SAVE_DIR}'):
    os.makedirs(f'{MODEL_SAVE_DIR}')

if not os.path.isdir(f'{GRAPH_SAVE_DIR}'):
    os.makedirs(f'{GRAPH_SAVE_DIR}')

train_scores = np.zeros((EPOCHS))
val_scores = np.zeros((EPOCHS))
train_losses = np.zeros((EPOCHS))
val_losses = np.zeros((EPOCHS))
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, preds = evaluate(model, device, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print("|Epoch: {0:d} | Train loss: {1:.3f} | Train Acc: {2:.2f}% | Val Loss : {3:.3f} | Val Acc: {4:.2f}% | Time used: {5:d}s |".format(epoch+1, 
        train_loss, train_acc*100, valid_loss, valid_acc*100, int(time.time()-time_curr)))

    time_curr = time.time()

    train_scores[epoch] = train_acc
    val_scores[epoch] = valid_acc

    train_losses[epoch] = train_loss
    val_losses[epoch] = valid_loss

X = np.linspace(0, EPOCHS, EPOCHS)
visualize(X, train_scores, val_scores, train_losses, val_losses, f"{GRAPH_SAVE_DIR}/learning")

### Testing ###
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss, test_acc, preds = evaluate(model, device, test_iterator, criterion)
y_true = np.array(torch.cat([torch.round(y) for _,y in test_iterator]), dtype=int)
visualize_testing(y_true,preds,f'{GRAPH_SAVE_DIR}/h_eval')

print('| Test Loss: {0:.3f} | Test Acc: {1:.3f}% |'.format(test_loss,test_acc*100))

print('\nPrediction on Mouse data')
test_loss, test_acc, preds = evaluate(model, device, mouse_train_iterator, criterion)
print('| Test Loss: {0:.3f} | Test Acc: {1:.3f}% |'.format(test_loss,test_acc*100))

y_true = np.array(torch.cat([torch.round(y) for _,y in mouse_train_iterator]), dtype=int)
visualize_testing(y_true,preds,f'{GRAPH_SAVE_DIR}/m_eval')
