import os
import numpy as np
import pandas as pd
import nibabel as nib


# Substitute the "parent_path" variable with your own.
# The parent_path should contain the three subfolders:
# 1) AgeSex
# 2) UAS_dataset
# 3) KS_dataset

parent_path = "/media/bill/12EB-1B9D/"


# Age, Sex data to be appended to the volumetric data.
agesex_uas_controls = pd.read_excel(parent_path + "AgeSex/AgeSex_UAS_controls_final.xlsx")
agesex_uas_nph = pd.read_excel(parent_path + "AgeSex/AgeSex_UAS_NPH_final.xlsx")
agesex_ks_controls = pd.read_excel(parent_path + "AgeSex/AgeSex_KS_controls_final.xlsx")
agesex_ks_nph = pd.read_csv(parent_path + "AgeSex/AgeSex_KS_NPH_final.csv", delimiter=";")

gender = {"F": 0, "M": 1}


paths_dict = {
    parent_path + "UAS_dataset/UAS_controls_FS_SS/UAS_controls_FS_output/" :
    [parent_path + "UAS_dataset/UAS_controls_FS_SS/UAS_controls_SS_output/", "_mask.mgz", agesex_uas_controls, 1],

    parent_path + "UAS_dataset/UAS_NPH_FS_SS/UAS_NPH_FS_output/" :
    [parent_path + "UAS_dataset/UAS_NPH_FS_SS/UAS_NPH_SS_output/", "_mask.mgz", agesex_uas_nph, 2],

    parent_path + "KS_dataset/KS_controls_FS_SS/KS_controls_FS_output/" :
    [parent_path + "KS_dataset/KS_controls_FS_SS/KS_controls_SS_output/", ".mgz", agesex_ks_controls, 3],

    parent_path + "KS_dataset/KS_NPH_FS_SS/KS_NPH_FS_output/" :
    [parent_path + "KS_dataset/KS_NPH_FS_SS/KS_NPH_SS_output/", ".mgz", agesex_ks_nph, 4]
}


# Create the DataFrame.
# It combines the FastSurfer volumetric data
# , with the age, sex data from above.
E=[]

ids=[]
labels=[]
flag_p_plus=False

for path in paths_dict:
    # print(path)
    
    for subj in os.listdir(path):
        print(subj)
        ids.append(subj)
        if paths_dict[path][3] == 1:
            tag = subj
        if paths_dict[path][3] == 2:
            if "ASMRX12" in subj:
                tag = "impr"+subj
            else:
                tag = subj
        if paths_dict[path][3] == 3:
            tag = "contr"+subj
        if paths_dict[path][3] == 4:
            tag = "impr"+subj
            
        # print(tag)
        row=[]
        with open(path + subj + "/stats/aseg+DKT.stats", "r") as file:
            f = file.read()
            l = f.split("#")[-1].split("\n")[1:-1]
            g = f.split("#")[22].split()
            for e in l:
                row.append( float(e.split()[3]) )
            a = nib.load(f"{paths_dict[path][0]}TIVmasks/{subj}{paths_dict[path][1]}").get_fdata()
            voxs = a.sum()
            if g[0] == "VoxelVolume_mm3":
                row.append( float(g[1]) * voxs )
            else:
                print("VoxelVolume_Error !")
        agesex = paths_dict[path][2]
        if "ASMRX12" not in subj:
            s = float(subj.split("_")[-2])
            age = float( list(agesex[agesex["ID"]==s]["Age"])[0] )
            sex = gender[ list(agesex[agesex["ID"]==s]["Sex"])[0] ]
        if "ASMRX12" in subj:
            s = subj[0:-4]
            age = float( list(agesex[agesex["ID"]==s]["Age"])[0] )
            sex = gender[ list(agesex[agesex["ID"]==s]["Sex"])[0] ]
        row.append(age)
        row.append(sex)
        
        if "_controls_" in path and (age >= 62):
            row.append(0)
            diagnosis = list(agesex[agesex["ID"]==s]["Diagnosis"])[0]
            if diagnosis == "CBS or MSA":
                diagnosis = "MSA-P"
            if diagnosis == "P+":
                if flag_p_plus==False:
                    diagnosis = "MSA-P"
                    flag_p_plus = True
                elif flag_p_plus == True:
                    diagnosis = "PSP"
            E.append(row)
            labels.append(diagnosis)
            
        if "_NPH_" in path:
            row.append(1)
            diagnosis = "iNPH"
            E.append(row)
            labels.append(diagnosis)

data = pd.DataFrame(E)


# Normalization with Total IntraCranial Volume
# and deletion of CC (Corpus Callosum).
p = parent_path + "UAS_dataset/UAS_NPH_FS_SS/UAS_NPH_FS_output/ASMRX12_U1_seg/stats/"
n = "aseg+DKT.stats"
E = []

with open(p+n, "r") as file:
    cols = []
    # cols.append("ID") ### add/remove for ID name of subject column
    l = file.read().split("#")[-1].split("\n")[1:-1]
    for e in l:
        #row.append( float(e.split()[3]) )
        cols.append( e.split()[4] )
    #E.append( row.copy() )
    cols.append("TIV")
    cols.append("Age")
    cols.append("Sex")
    cols.append("iNPH")

data.columns = cols
data = data.drop(["CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior"], axis=1)
data.iloc[:, 0:-4] = data.iloc[:, 0:-4].div(data["TIV"], axis=0)
data = data.drop( "TIV", axis=1 )


labels = pd.DataFrame(labels)


# export to csv:
data.to_csv("./volumetric_data.csv")
labels.to_csv("./volumetric_labels.csv")

# data is the .csv file with the actual volumetric+age+sex data

# labels is the .csv file with the subcategories of the control group
#   ,used only for split stratification purpose.
