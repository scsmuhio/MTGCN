#!/usr/bin/env python
# coding: utf-8

# In[22]:


#from node2vec import Node2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from polyglot_tokenizer import Tokenizer
from wxconv import WXC
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
#get_ipython().run_line_magic('matplotlib', 'inline')
from indicnlp.tokenize import indic_tokenize 
con = WXC(order='utf2wx',lang='tel')
#tk = Tokenizer(lang='te', split_sen=True)

# In[2]:


data = pd.read_excel('./DATA.xlsx', header = None)
data.head()


# In[3]:


data = data[data[1]!='neutral']


# In[36]:


#df = data.copy()


# In[4]:


print(data.head())


# In[5]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data[1])
sen_labels = enc.transform(data[1])


# In[39]:


from indicnlp.tokenize import indic_tokenize  
con = WXC(order='utf2wx',lang='tel')
def create_vocab(sentences):
  vocab = []
  x = []
  for each_sent in sentences:
    x.append(indic_tokenize.trivial_tokenize(con.convert(each_sent)))
    for t in indic_tokenize.trivial_tokenize(con.convert(each_sent)):
      vocab.append(t)
  
  return vocab,x


# In[40]:


vocab, tokens = create_vocab(data[0])  
print(len(vocab))

vocab = set(vocab)
print(len(vocab))


# In[41]:


print(tokens[0])


# In[42]:


from collections import Counter
from itertools import combinations
cx = Counter()
cxy = Counter()
for text in tokens:
    
    for x in text:
        cx[x] += 1

    # Count all pairs of words, even duplicate pairs.
    for x, y in map(sorted, combinations(text, 2)):
        cxy[(x, y)] += 1


# In[43]:


print(len(data))


# In[44]:


x2i, i2x = {}, {}
for i, x in enumerate(cx.keys()):
    x2i[x] = i
    i2x[i] = x


# In[45]:
'''

sx = sum(cx.values())
sxy = sum(cxy.values())


# In[46]:
idf = Counter()
for x in cx:
    for text in tokens:
        if x in text:
            idf[x]+=1

tf_idf = Counter()
j=41501
for text in tokens:
    try:
        for x in text:
            tf_idf[(x, j)] = text.count(x)*np.log(data.shape[0]/idf[x])
        j+=1
    except:
        continue

from math import log
pmi_samples = Counter()
data1, rows, cols = [], [], []
for (x, y), n in cxy.items():
    rows.append(x2i[x])
    cols.append(x2i[y])
    data1.append(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))
    pmi_samples[(x, y)] = data1[-1]
#PMI = csc_matrix((data1, (rows, cols)),dtype=np.int8).toarray()
for (x, y), n in tf_idf.items():
    rows.append(x2i[x])
    cols.append(y)
    #data1.append(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))
    data1.append(tf_idf[(x,y)])
    pmi_samples[(x, y)] = tf_idf[(x,y)]
for (x, y), n in tf_idf.items():
    rows.append(y)
    cols.append(x2i[x])
    #data1.append(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))
    data1.append(tf_idf[(x,y)])
    pmi_samples[(y, x)] = tf_idf[(x,y)]
PMI = csc_matrix((data1, (rows, cols))).toarray()
PMI = np.array(PMI, dtype=np.float32)
'''
# In[47]:

import torch
#print(PMI.shape)
#np.save('PMI', PMI)
PMI = np.load('PMI.npy')
print(PMI.shape)
# In[48]:
indices = list(range(20000))+list(range(41501,57735))
ixgrid = np.ix_(indices, indices)
PMI = PMI[ixgrid]
print(PMI.shape)
#exit(0)
#PMI = PMI[20000:,20000:]
PMI = torch.Tensor(PMI)

#PMI = PMI[0:22000, 0:22000]


#def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)
#PMI = sparse_mx_to_torch_sparse_tensor(PMI)

# In[49]:


#from pprint import pformat
#print('%d non-zero elements' % PMI.count_nonzero())
#print('Sample PMI values\n', pformat(pmi_samples.most_common()[:10]))

'''
# In[6]:


PMI = np.load('D:/Telugu_Node2vec/word_coocc_indic.npy')
vocab = np.load('D:/Telugu_Node2vec/word_vocab_indic3.npy')
features = np.load('./word_vocab_indic3.npy')

print(vocab.shape)
print(features.shape)
#exit(0)
# In[20]:

features = features[22000:]
print(features.shape)
#exit(0)
PMI = PMI[22000:, 22000:]
print(PMI.shape)

x2i, i2x = {}, {}
for i, x in enumerate(vocab):
    x2i[x] = i
    i2x[i] = x

#print(x2i)
#exit(0)
# In[7]:
'''

#from __future__ import division
#from __future__ import print_function
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
import os
import networkx as nx
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
import scipy
import torch
# from scipy import stats
from sklearn.model_selection import LeaveOneOut,KFold, train_test_split
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

import time
import argparse
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy
from models_telugu import GCN
# import EarlyStopping
from pytorchtools import EarlyStopping
from matplotlib.ticker import FuncFormatter, MaxNLocator
#from pygsp import graphs, filters
#from pygkernels.cluster import KKMeans
#from pygkernels.measure import logComm_H
#from torch_sparse import SparseTensor

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from transformers import pipeline
import lightgbm as lgb
#import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# In[9]:


def LR(xtrain,xtest,ytrain,ytest):
    log_model = LogisticRegression()
    log_model.fit(xtrain, ytrain)
    y_pred_val = log_model.predict(xtest)
    y_pred_train = log_model.predict(xtrain)
    return y_pred_train, y_pred_val
    #print(confusion_matrix(ytest, y_pred_log))
    #print(classification_report(ytest,y_pred_log))
def MLP(xtrain,xtest,ytrain,ytest):
    log_model = xgb.XGBClassifier()
    log_model.fit(xtrain, ytrain)
    y_pred_log = log_model.predict(xtest)
    print(confusion_matrix(ytest, y_pred_log))
    print(classification_report(ytest,y_pred_log))
def LGBM(xtrain,xtest,ytrain,ytest):
    lgbm_model = lgb.LGBMClassifier(n_estimators=1000,max_depth=5)
    lgbm_model.fit(xtrain, ytrain)
    #y_pred_lgbm = lgbm_model.predict(xtest)
    y_pred_val = lgbm_model.predict(xtest)
    y_pred_train = lgbm_model.predict(xtrain)
    return y_pred_train, y_pred_val
    #print(confusion_matrix(ytest, y_pred_lgbm))
    #print(classification_report(ytest,y_pred_lgbm))

#aa = []
#for i in np.arange(features.shape[0]):
#	aa.append(features[i])
#aa = np.array(aa)
#print(aa.shape)
#features = torch.Tensor(features)
#print(features.shape)

# In[10]:


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=15,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batchsize', type=int, default=7,
                    help='Batch Size')


# In[11]:


args = parser.parse_args([])
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
#if args.cuda:
#    torch.cuda.manual_seed(args.seed)

Path = './GCN.pth.tar'
global counter 
counter = 0


# In[12]:


np.random.seed(args.seed)
torch.manual_seed(args.seed)


# In[56]:


def generate_features(data,features):
    text_features = []
    for i in data:
        temp = []
        tokens = indic_tokenize.trivial_tokenize(con.convert(i))
        for token in tokens:
            try:
                #print(features[x2i[token]].shape)
                temp.append(features[x2i[token]])
            except:
                temp.append(np.zeros([1,200]))
                continue
        temp = np.array(temp)
        temp = np.mean(temp,axis=0)
        temp = np.reshape(temp, (1, 200))
        text_features.append(temp)
    text_features = np.array(text_features)
    #print(text_features.shape)
    text_features = np.reshape(text_features,(text_features.shape[0], text_features.shape[2]))
    return text_features


#if args.cuda:
#    PMI = torch.Tensor(PMI).cuda()
# In[16]:
'''

def generate_features(data,features):
    text_features = []
    for i in data:
        temp = []
        #tokens = indic_tokenize.trivial_tokenize(i)
        tokens = tk.tokenize(i)
        for token in tokens[0]:
            try:
                #print(features[x2i[token]].shape)
                temp.append(features[x2i[token]])
            except:
                temp.append(np.zeros([1,200]))
                continue
        temp = np.array(temp)
        temp = np.mean(temp,axis=0)
        temp = np.reshape(temp, (1, 200))
        text_features.append(temp)
    text_features = np.array(text_features)
    #print(text_features.shape)
    text_features = np.reshape(text_features,(text_features.shape[0], text_features.shape[2]))
    return text_features

'''
# In[17]:

loss = []
lossval = []
#valloss = np.array([])
def train(model,epoch,trainindex, valindex):
#     loss = []
    temp = []
    temp_val = []
    t = time.time()
    for i in np.arange(1): 
        SC_data_u = torch.eye(PMI.shape[0])
        #SC_data_u = features
        model.train()
        optimizer.zero_grad()
        output,latent = model(SC_data_u, PMI)
        #print('Output shape', str(latent.shape))
        train_features = generate_features(data[0].iloc[trainindex].values, latent.detach().numpy()[0:20000,0:20000])
        val_features = generate_features(data[0].iloc[valindex].values, latent.detach().numpy()[0:20000,0:20000])
        #print(train_features.shape, val_features.shape)
        y_pred_train,y_pred_val = LR(train_features,val_features,sen_labels[trainindex],sen_labels[valindex])
        #print('Output shape', str(output.shape))
        loss_train = F.mse_loss(output, torch.Tensor(PMI)) + F.binary_cross_entropy(torch.Tensor(sen_labels[trainindex]), torch.Tensor(y_pred_train))
        loss_train.backward()
        optimizer.step()
        temp.append(loss_train.detach().numpy())
    loss.append(np.mean(np.array(temp)))
    if not args.fastmode:
        model.eval()
        for j in np.arange(1):
            #output = model(torch.eye(rows), torch.from_numpy(L_sc[j]).float())
            output,latent = model(SC_data_u, PMI)
            #np.save('word_graph',output.detach().numpy())
            np.save('MT-GCN200_Start',latent.detach().numpy())
            #print('Output shape', str(latent.shape))
        train_features = generate_features(data[0].iloc[trainindex].values, latent.detach().numpy()[0:20000,0:20000])
        val_features = generate_features(data[0].iloc[valindex].values, latent.detach().numpy()[0:20000,0:20000])
        print(train_features.shape, val_features.shape)
        y_pred_train, y_pred_val = LR(train_features,val_features,sen_labels[trainindex],sen_labels[valindex])
        #print('Output shape', str(output.shape))
        loss_val = F.mse_loss(output, torch.Tensor(PMI)) + F.binary_cross_entropy(torch.Tensor(sen_labels[valindex]), torch.Tensor(y_pred_val))
        temp_val.append(loss_val.detach().numpy())
        lossval.append(np.mean(np.array(temp_val)))
        valloss = np.mean(np.array(temp_val))
        print('Epoch: {:04d}'.format(epoch+1),
        #f'Batch:{(i+1)/args.batchsize}/{examples/args.batchsize}',
        'loss_train: {:.9f}'.format(loss_train.item()),
        'loss_val: {:.5f}'.format(np.mean(np.array(temp_val))),
        'time: {:.4f}s'.format(time.time() - t))
        print(confusion_matrix(sen_labels[testindex], y_pred_val))
        print(classification_report(sen_labels[testindex],y_pred_val))
    # torch.save(model.state_dict(),Path+'model.txt')
    #torch.save(model.state_dict(),Path)
    #model.load_state_dict(torch.load('checkpoint.pt'))
    return valloss
    #model.load_state_dict(torch.load('checkpoint.pt'))


# In[18]:


import os
correlations = []
mse = []
losstest = []
def test(model,testindex):
    #testindex = np.delete(np.arange(98), rand_indeces)
    for i in np.arange(1):
        model.eval()
        output,latent = model(torch.eye(PMI.shape[0]), torch.Tensor(PMI))
        #print('Output shape', str(latent.shape))
        train_features = generate_features(data[0].iloc[trainindex].values, latent.detach().numpy()[0:41501,0:41501])
        test_features = generate_features(data[0].iloc[testindex].values, latent.detach().numpy()[0:41501,0:41501])
        print(train_features.shape, test_features.shape)
        #text_labels = sen_labels[trainindex][np.where(train_features.any(axis=1))[0]]
        #train_features = train_features[np.where(train_features.any(axis=1))[0]]
        #print(train_features.shape, text_labels.shape)
        #break
        y_pred_train, y_pred_test = LR(train_features,test_features,sen_labels[trainindex],sen_labels[testindex])
        #print('Output shape', str(output.shape))
        loss_test = F.mse_loss(output, torch.Tensor(PMI)) + F.binary_cross_entropy(torch.Tensor(sen_labels[testindex]), torch.Tensor(y_pred_test))
        print(loss_test)
        losstest.append(loss_test)
        print(confusion_matrix(sen_labels[testindex], y_pred_test))
        print(classification_report(sen_labels[testindex],y_pred_test))


# In[23]:


since = time.time()
#loo_c = LeaveOneOut()
corr_val = []
loss_val = []
#loo = loo_c.split(np.arange(100))
kfold = KFold(n_splits=5, shuffle=True)
print(kfold)
for trainindex, testindex in kfold.split(np.arange(data.shape[0])):
    trindex, valindex = train_test_split(trainindex, test_size=0.1)
    model = GCN(nfeat=PMI.shape[0],
            nhid=200,
            nclass=PMI.shape[0],
            dropout=args.dropout)
    #model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(100):
        #if os.path.exists(Path):
        #    model.load_state_dict(torch.load(Path))
        #model.load_state_dict(torch.load('SC_FC_connection/checkpoint.pt'))
        valloss = train(model,epoch,trindex,testindex)
        print(valloss)
        early_stopping(valloss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        #counter += 1
        #break
        
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - since))
    #test(model,testindex)
    #os.remove(Path)
    break


# In[63]:

np.save('train_loss',loss)
np.save('test_loss',lossval)

fig,ax = plt.subplots(figsize=(10,6))
# Edit- Below line commented because there was no output for loss[300:400]
plt.plot(np.array(loss), linewidth=2)
plt.plot(lossval, linewidth=2)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Epochs vs Loss')
plt.xlabel('epochs', fontsize=18)
plt.ylabel('loss', fontsize=18)
plt.legend(['training loss','validation loss'])
#plt.plot(loss)
plt.show()


# In[71]:


test(model,valindex)


# In[73]:


test(model,testindex)


# In[13]:


#vocab[0]


# In[15]:


#list(vocab).index('\xa0భద్రతకు')


# In[ ]:




