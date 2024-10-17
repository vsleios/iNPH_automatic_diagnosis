import numpy as np
import pandas as pd
from sklearn import metrics
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch import nn
from torch import Tensor

from rtdl_revisiting_models import ResNet


########################### enabling GPU
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
       
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#################################################

################ loading the files: volumetric_data.csv, volumetric_labels.csv
data = pd.read_csv("./volumetric_data.csv", index_col=0)

labels = pd.read_csv("./volumetric_labels.csv", index_col=0)
labels = list(labels.iloc[:,0])


from torch.utils.data import Dataset

class iNPH_Dataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, index):
        # a = torch.from_numpy(nib.load(self.df.iloc[index,0]).get_fdata())
        return ( torch.from_numpy(np.array(self.df.iloc[index,0:-1])) , self.df.iloc[index,-1] )


######### hyperparameters of the ResNet model. 
N_BLOCKS = [2,3]
D_BLOCK = [15, 30]
DROPOUT = [0.0, 0.2]

# model = ResNet (
#         d_in = 97,
#         d_out=1,
#         n_blocks=3,
#         d_block=15,
#         d_hidden=None,
#         d_hidden_multiplier=2.0,
#         dropout1=0.0,
#         dropout2=0.0,
#     ).to(device)

GRID = []

for n in N_BLOCKS:
    for d in D_BLOCK:
        for dr in DROPOUT:
            GRID.append( [n, d, dr] )


def NN_pipeline(m, trainloader, testloader, path): # path = False or path = {path}.
    torch.cuda.empty_cache()
    model=m
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    
    # train_acc=[]; test_acc=[]
    acc_max_pipeline = -1
    epoch=0; patience=0
    while epoch <= 2000 and patience <= 2000:
        epoch += 1
        for x, y in trainloader:
            model.train()
            x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
            pred = model(x).squeeze()
            pred = torch.sigmoid(pred)
            loss_0 = loss_fn(pred, y)
                    
            optimizer.zero_grad()
            loss_0.backward()
            optimizer.step()
                
            # q = (pred>0.5).to(torch.float32)
            # t = torch.mean((y == q).to(torch.float32)).item()
            # train_acc.append(t)
    
        for x, y in testloader:
            model.eval()
            x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
            pred = model(x).squeeze()
            pred = torch.sigmoid(pred)
            loss_0 = loss_fn(pred, y)
                            
            q = (pred>0.5).to(torch.float32)
            t = torch.mean((y == q).to(torch.float32)).item()
            # test_acc.append(t)
            if t > acc_max_pipeline:
                patience = 0
                acc_max_pipeline = t
                if path != False:
                    model.eval()
                    torch.save(model.state_dict(), path)
                max_epoch = epoch
            else:
                patience += 1
            
    for x, y in testloader:
        model.eval()
        x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
        pred = model(x).squeeze()
        pred = torch.sigmoid(pred)
        loss_0 = loss_fn(pred, y)
        
        q = (pred>0.5).to(torch.float32)                   
        ac, se, sp, f1 = ss(q.cpu().detach().numpy(), y.cpu().detach().numpy())
        print(ac, se, sp, f1)
        
    # print(max_epoch)
    return (ac, se, sp, f1)

#############################sensitivity specificity function + Confidence Interval + F1-score
from statsmodels.stats.proportion import proportion_confint
def ss( pred, truth ):

    z = [ [pred[i], truth[i]] for i in range(len(truth)) ]
    n = len(pred)

    acc = np.mean([ pred[i]==truth[i] for i in range(len(pred)) ])

    sens = z.count([1,1]) / ( z.count([1,1]) + z.count([0,1]) )
    
    spec = z.count([0,0]) / ( z.count([0,0]) + z.count([1,0]) )

    rec = sens
    if np.sum(pred)==0:
        print(" ##########################  WARNING  ######################## predicted all 0.")
        pred[0]=1
        z = [ [pred[i], truth[i]] for i in range(len(truth)) ]
        rec = z.count([0,0]) / ( z.count([0,0]) + z.count([1,0]) )
    pre = z.count([1,1]) / ( z.count([1,1]) + z.count([1,0]) )

    if pre+rec == 0:
        print("  ######## ########## ########## WARNING ########### ######### ########## f1 not defined -> return 0")
        f1 = 0
    else:
        f1 = (2*pre*rec) / (pre + rec)

    return (acc, sens, spec, f1)


### Nested Cross Validation
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

NCV=[] # Nested Cross Validation: keeps the outer hyperparameters with their performances.

kf_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
k=0

for outer_train_index, outer_test_index in kf_out.split(X=data, y=labels):
    print("OUTER ", k)
    outer_train = data.iloc[outer_train_index, :]
    outer_test = data.iloc[outer_test_index, :]

    outer_max_acc=-1 # keep the best accuracy of the inner loop
    hyper=[] # keep the best hyperparameters of the inner loop
    for comb in GRID:
        cv_acc=[]
        kf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
        for inner_train_index, inner_test_index in kf_in.split(X=outer_train, y=[ labels[i] for i in outer_train_index ]):
            # print("INNER")
            inner_train = outer_train.iloc[inner_train_index, :]
            inner_test = outer_train.iloc[inner_test_index, :]
        
            train_dataset = iNPH_Dataset(inner_train)
            train_dataloader = DataLoader(train_dataset, batch_size=inner_train.shape[0], shuffle=True)
                    
            test_dataset = iNPH_Dataset(inner_test)
            test_dataloader = DataLoader(test_dataset, batch_size=inner_test.shape[0], shuffle=False)
                
            #model initialization:
            model = ResNet (d_in = 97,d_out=1,n_blocks=comb[0],d_block=comb[1],
                            d_hidden=None,d_hidden_multiplier=2.0,dropout1=comb[2],dropout2=0.0,).to(device)
            
            acc, sen, spe, f1 = NN_pipeline(m=model, trainloader=train_dataloader, testloader=test_dataloader, path=False)
            cv_acc.append(acc)
            del model
                
        accuracy = np.mean(cv_acc)
        if accuracy >= outer_max_acc:
            outer_max_acc = accuracy
            hyper = comb
    torch.cuda.empty_cache()
    # hyper: has kept the optimal hyperparameter combination
    print(f"optimal hyperparameters {hyper}: {outer_max_acc}")
    print("Retrain to outer train set:")
    train_dataset = iNPH_Dataset(outer_train)
    train_dataloader = DataLoader(train_dataset, batch_size=outer_train.shape[0], shuffle=True)
            
    test_dataset = iNPH_Dataset(outer_test)
    test_dataloader = DataLoader(test_dataset, batch_size=outer_test.shape[0], shuffle=False)

    #model initialization for retraining (using optimal hyperparameters):
    model = ResNet (d_in = 97,d_out=1,n_blocks=hyper[0],d_block=hyper[1],
                            d_hidden=None,d_hidden_multiplier=2.0,dropout1=hyper[2],dropout2=0.0,).to(device)
    
    acc, sen, spe, f1 = NN_pipeline(m=model, trainloader=train_dataloader, testloader=test_dataloader, path=f"./RES{k}.pth")
    NCV.append([hyper[0],hyper[1], hyper[2], acc, sen, spe, f1 ])
    k += 1

ncv = pd.DataFrame(NCV)
ncv.to_csv("./RESncv.csv")

print("finished.")
