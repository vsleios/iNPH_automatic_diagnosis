import numpy as np
import pandas as pd
from sklearn import metrics
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch import nn
from torch import Tensor

from rtdl_revisiting_models import FTTransformer


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


### model architecture:
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

class FC2 (nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        
        self.fc1 = nn.Linear(97,70)
        self.fc2 = nn.Linear(70, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 1)

        self.summary = "FC-dropout: 0.1"
    
#gaussian weights and offsets:

    # def init_weights(self):
    #         for layer in self.modules():
    #             if isinstance(layer, nn.Linear):
    #                 init.normal_(layer.weight, mean=0, std=0.001)
    #                 if layer.bias is not None:
    #                     init.normal_(layer.bias, mean=0, std=0.001)

        
    def forward(self, x):
        #x: 2x128x128
        
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)

        x = self.fc5(x)
        
        x = torch.sigmoid(x)
        return(x)


## Helper function for training the TRansformer model
def NN_pipeline(m, trainloader, testloader, path): # path = False or path = {path}.
    torch.cuda.empty_cache()
    model=m
    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    
    # train_acc=[]; test_acc=[]
    acc_max_pipeline = -1
    epoch=0; patience=0
    while epoch <= 2000 and patience <= 500:
        epoch += 1
        for x, y in trainloader:
            model.train()
            x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
            pred = model(x, None).squeeze()
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
            pred = model(x, None).squeeze()
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
        pred = model(x, None).squeeze()
        pred = torch.sigmoid(pred)
        loss_0 = loss_fn(pred, y)
        
        q = (pred>0.5).to(torch.float32)                   
        ac, se, sp, f1 = ss(q.cpu().detach().numpy(), y.cpu().detach().numpy())

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


### Nested Cross Validation - retraining variation:
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

kf_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
k=0
ACC=[]; SEN=[]; SPE=[]; F1=[]
for outer_train_index, outer_test_index in kf_out.split(X=data, y=labels):
    print("OUTER ", k)
    outer_train = data.iloc[outer_train_index, :]
    outer_test = data.iloc[outer_test_index, :]

    kf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    inner_max_acc = -1
    for inner_train_index, inner_test_index in kf_in.split(X=outer_train, y=[ labels[i] for i in outer_train_index ]):
        print("INNER")
        inner_train = outer_train.iloc[inner_train_index, :]
        inner_test = outer_train.iloc[inner_test_index, :]

        model = FC2().to(device)

        train_dataset = iNPH_Dataset(inner_train)
        train_dataloader = DataLoader(train_dataset, batch_size=inner_train.shape[0], shuffle=True)
            
        test_dataset = iNPH_Dataset(inner_test)
        test_dataloader = DataLoader(test_dataset, batch_size=inner_test.shape[0], shuffle=False)
        
        acc, sen, spe, f1 = NN_pipeline(m=model, trainloader=train_dataloader, testloader=test_dataloader, lr=0.001)

        if acc >= inner_max_acc:
            inner_max_acc = acc
            model.eval()
            torch.save(model.state_dict(), "./NCV/inner_model/vnn_inner.pth")

    print("Retrain to outer train set:")
    train_dataset = iNPH_Dataset(outer_train)
    train_dataloader = DataLoader(train_dataset, batch_size=outer_train.shape[0], shuffle=True)
            
    test_dataset = iNPH_Dataset(outer_test)
    test_dataloader = DataLoader(test_dataset, batch_size=outer_test.shape[0], shuffle=False)

    model = FC2().to(device)
    model.load_state_dict(torch.load( "./NCV/inner_model/vnn_inner.pth" ))
    acc, sen, spe, f1 = NN_pipeline(m=model, trainloader=train_dataloader, testloader=test_dataloader, lr=0.001)
    
    model.eval()
    torch.save(model.state_dict(), f"./NCV/outer_models/vnn{k}.pth")
    
    ACC.append(acc); SEN.append(sen); SPE.append(spe); F1.append(f1)
    k += 1
print(ACC); print(SEN); print(SPE); print(F1)



### Retrain the VNN to the whole Dataset:
all_dataset = iNPH_Dataset(data)
all_dataloader = DataLoader(all_dataset, batch_size=data.shape[0], shuffle=True)

model = FC2().to(device)
model.load_state_dict(torch.load( "./NCV/outer_models/vnn3.pth" ))
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.00001)
loss_fn = nn.BCELoss()

train_acc=[]; test_acc=[]
for epoch in range(2000):
    for x, y in all_dataloader:
        model.train()
        x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
                # print(x.shape, y.shape)
                # print(x[0,:])
        pred = model(x).squeeze()
                # print(pred)
        loss_0 = loss_fn(pred, y)
                # print(loss_0.item())
                
        optimizer.zero_grad()
        loss_0.backward()
        optimizer.step()
                
                # fpr, tpr, thresh = metrics.roc_curve(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), pos_label=1)
        q = (pred>0.5).to(torch.float32)
                
                # a1,b1 = ss0(q, y)
        t = torch.mean((y == q).to(torch.float32)).item()
                # print(t)
        train_acc.append(t)

    for x, y in all_dataloader:
        model.eval()
        x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
                        # print(x.shape, y.shape)
                        # print(x[0,:])
        pred = model(x).squeeze()
                        # print(pred)
        loss_0 = loss_fn(pred, y)
                        # print(loss_0.item())
                        
                        # fpr, tpr, thresh = metrics.roc_curve(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), pos_label=1)
        q = (pred>0.5).to(torch.float32)
                        
        # a1,b1 = ss0(q, y)
        t = torch.mean((y == q).to(torch.float32)).item()
                    # a1,b1 = ss0(q, y)
                    # print(t)
        test_acc.append(t)
        
### overal "train" performance
for x, y in all_dataloader:
    model.eval()
    x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
                    # print(x.shape, y.shape)
                    # print(x[0,:])
    pred = model(x).squeeze()
                    # print(pred)
    loss_0 = loss_fn(pred, y)
                    # print(loss_0.item())
                    
                    # fpr, tpr, thresh = metrics.roc_curve(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), pos_label=1)
    q = (pred>0.5).to(torch.float32)
                    
    a1,b1 = ss0(q, y)
    t = torch.mean((y == q).to(torch.float32)).item()
                # a1,b1 = ss0(q, y)
                # print(t)
    print(t, a1, b1)
    
plt.plot(train_acc, color="blue", alpha=1, label="Train")
plt.plot(test_acc, color="red", alpha=0.3, label="Test")
plt.title("Train Test Accuracies")
plt.legend()
plt.show()


### save the model:
model.eval()
torch.save(model.state_dict(), "./VNN_final.pth")

