from matplotlib import pyplot as plt
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

################ loading the file: volumetric_data.csv
data = pd.read_csv("./volumetric_data.csv", index_col=0)


# custom dataset:
from torch.utils.data import Dataset
class iNPH_Dataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, index):
        # a = torch.from_numpy(nib.load(self.df.iloc[index,0]).get_fdata())
        return ( torch.from_numpy(np.array(self.df.iloc[index,0:-1])) , self.df.iloc[index,-1] )



## helper function for the trichotomic performance
def cutoffs(a, b, p, cut):
    
    a1=[]; b1=[]
    for i in range(1,100):
        if np.max(a[i:])<=cut:
            a1.append(i)
        if np.max(b[:i])<=cut:
            b1.append(i)
    x=np.min(a1)/100; y=np.max(b1)/100
    
    c=0; c2=0
    for i in range(len(p)):
        if p[i] > x and p[i] < y:
            c += 1
        else:
            c2 += 1

    print(x, y, "percentage of points that fall in the intermediate bin : ", round(c/(c+c2), 3) )
    return( x, y, c/(c+c2) )


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


### Loading the model:
model = FTTransformer(
    n_cont_features=97,
    cat_cardinalities=[],
    d_out=1,
    
    # **FTTransformer.get_default_kwargs(),
    
    n_blocks=3,
    d_block=384,
    attention_n_heads=4,
    attention_dropout=0.2,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1,
    residual_dropout=0.0,

    # linformer_kv_compression_ratio=0.2,           
    # linformer_kv_compression_sharing='headwise'
    
).to(device)

model.load_state_dict(torch.load( "./FTT_final.pth" ))
model.eval()


## Trichotomic performance
from torch.utils.data import DataLoader
all_dataset = iNPH_Dataset(data)
all_dataloader = DataLoader(all_dataset, batch_size=data.shape[0], shuffle=True)
    
for x, y in all_dataloader:
            
    x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
    pred = model(x, None).squeeze()
    pred = torch.sigmoid(pred)
    q = (pred>0.5).to(torch.float32)
    # _, _, a, _, _, b, _, _, c = ss(q.cpu().detach().numpy(), y.cpu().detach().numpy())
    # t = torch.mean((y == q).to(torch.float32)).item()
    # print(t, a, b, c)
    r = ss(q.cpu().detach().numpy(), y.cpu().detach().numpy())
    print(r)

    #####
    print("\n98% Triadic performance trained on the whole dataset :\n")
    cuts = np.cumsum( [0.01 for i in range(100)] )
    y=y.cpu().detach().numpy()
    p = pred.cpu().detach().numpy()
        
    l1=[]
    for c in cuts:
        q1=[]; y1=[]
        for i in range(len(p)):
            if p[i] < c:
                q1.append(0)
                y1.append(y[i])
        q1 = np.array(q1); y1 = np.array(y1)
        l1.append( np.mean(q1==y1) )
        
    l2=[]
    for c in cuts:
        q2=[]; y2=[]
        for i in range(len(p)):
            if p[i] > c:
                q2.append(1)
                y2.append(y[i])
        q2 = np.array(q2); y2 = np.array(y2)
        l2.append( np.mean(q2==y2) )
    l2[-1] = 1
    i=0
    while not (l1[i]>0.98 and l1[i+1]<0.98):
        i += 1
    x_1 = (0.98-l1[i])/(l1[i+1]-l1[i]) + i
    
    i=0
    while not (l2[i]<0.98 and l2[i+1]>0.98):
        i += 1
    x_2 = (0.98-l2[i])/(l2[i+1]-l2[i]) + i
    
    plt.plot(l1, zorder=0)
    plt.plot(l2, zorder=0)
    plt.axhline(y=0.98, c="gray", linewidth=1, linestyle="--")
    plt.scatter(x=[x_1, x_2], y=[0.98, 0.98], c="black", s=10)
    plt.axvline(x=x_1, c="black", linewidth=1, linestyle="--")
    plt.axvline(x=x_2, c="black", linewidth=1, linestyle="--")
    plt.xlabel("model probablity * 100")
    plt.ylabel("Negative/Positive Predictive Value")
    plt.savefig("./ftt_x.png")
    plt.show()
        
    c1, c2, c3 = cutoffs(l1, l2, p, 0.98)
