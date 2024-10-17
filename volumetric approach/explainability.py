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

## Loading the final FTT model:
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

## loading a split:

from torch.utils.data import DataLoader

train_dataset = iNPH_Dataset(train)
train_dataloader = DataLoader(train_dataset, batch_size=train.shape[0], shuffle=True)

test_dataset = iNPH_Dataset(test)
test_dataloader = DataLoader(test_dataset, batch_size=test.shape[0], shuffle=False)


# permutation method:
from random import shuffle
nn_imp={}
n=100

for i in range(test.shape[1]-1):
    print(test.columns[i])
    a=0
    for k in range(n):
        for x, y in test_dataloader:
            model.eval()
            l = [ e.item() for e in x[:,i] ]
            shuffle(l)
            for j in range(test.shape[0]):
                x[j,i] = l[j]
            x = x.to(torch.float32).to(device); y = y.to(torch.float32).to(device)
            p = model(x).squeeze()
            
            # fpr, tpr, thresh = metrics.roc_curve(y.cpu().detach().numpy(), p.cpu().detach().numpy(), pos_label=1)
            q = (p>0.5).to(torch.float32)
            # a,b = ss(q.squeeze(), x[:,train.shape[1]-1])
            t = torch.mean((y == q).to(torch.float32)).item()
            a += t
    nn_imp[test.columns[i]] = a/n

nn_imp = {k: v for k, v in sorted(nn_imp.items(), key=lambda item: item[1])}
nn_imp10 = { k: nn_imp[k] for k in list(nn_imp.keys())[0:25]  }
x_labels = list(nn_imp10.keys())
# plot feature importance
plt.bar(range(len(nn_imp10)), nn_imp10.values(), alpha=0.5)
# plt.xlabel("Structures sorted")
plt.ylabel("Accuracy")
plt.xticks(ticks=range(len(nn_imp10)), labels=nn_imp10.keys(), rotation=60, ha='right', rotation_mode='anchor')
# plt.tight_layout()
plt.savefig("./volperm.png", bbox_inches="tight")
plt.show()


## SHAP:
import shap
features = np.array(list(data.columns)[:-1])

def model_wrapper(data):
    data = np.array(data)
    data = torch.from_numpy(data).to(device).float()
    model.eval()
    with torch.no_grad():
        pred = model(data, None).squeeze()
        pred = torch.sigmoid(pred)
        return pred.cpu().numpy().flatten()

explainer = shap.KernelExplainer(model_wrapper, np.array(train.iloc[:,:-1]) )

shap_values = explainer.shap_values(test.iloc[:,:-1], nsamples=100)

shap.summary_plot(shap_values, test.iloc[:,:-1] ,feature_names=features, plot_type="dot", max_display=20)


