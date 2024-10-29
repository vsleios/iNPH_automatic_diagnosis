### inference mode for the FT-Transformer model

# 1) imports and configuration
# 2) loading the saved model (file should be saved in "./FTT_final.pth")
# 3) inference


# 1) IMPORTS AND CONFIGURATIONS
import sys
import numpy as np
import nibabel as nib
import torch
import warnings
warnings.filterwarnings("ignore")
from rtdl_revisiting_models import FTTransformer


##### age, sex
age = float(sys.argv[1])
sex = float(sys.argv[2])

########################### enabling GPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#########################################



# 2) LOADING THE SAVED MODEL:

model = FTTransformer(
    n_cont_features=97,
    cat_cardinalities=[],
    d_out=1,
    
    n_blocks=3,
    d_block=384,
    attention_n_heads=4,
    attention_dropout=0.2,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1,
    residual_dropout=0.0,
    
).to(device)

model.load_state_dict(torch.load( "./FTT_final.pth" ))
model.to(device)
model.eval()


# 3) INFERENCE

# stats file and synth-strip mask processing:
row=[]
with open("./aseg+DKT.stats", "r") as file:
    f = file.read()
    l = f.split("#")[-1].split("\n")[1:-1]
    g = f.split("#")[22].split()
    for e in l:
        row.append( float(e.split()[3]) )
    a = nib.load("./ASMRX13_contr_100_seg_mask.mgz").get_fdata()
    voxs = a.sum()
    if g[0] == "VoxelVolume_mm3":
        tiv = float(g[1]) * voxs
    else:
        print("VoxelVolume_Error !")

row = (np.delete(row, [33, 34, 35, 36, 37]) / tiv).tolist()

row.append(age)
row.append(sex)

row = np.array(row)


# applying the model:
x = torch.from_numpy(row).float().to(device)
x = x.unsqueeze(0)
with torch.no_grad():
    z = torch.sigmoid(model(x, None).squeeze())

print(f"The output probability is : {round(z.item(), 5)}")

