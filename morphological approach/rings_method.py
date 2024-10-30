import lapy
from lapy import diffgeo
from lapy import plot
from lapy import TriaMesh, Solver, plot, io
import numpy as np
import pandas as pd
import os; import shutil
import plotly
plotly.offline.init_notebook_mode(connected=True)
from matplotlib import pyplot as plt
from scipy import stats
import sys


# helper function
def smoothed(l, n): # we'll compute the moving average among the elements i-n, ..., i-1, i, i+1, ..., i+n
    a = [l[0]]*n
    b = [l[-1]]*n
    l2 = a+l+b
    r=[]
    for i in range(len(l)):
        r.append( np.mean(l2[i : i+2*n+1]) )
    return(r)

# defining a dictionary with the structures mapped to their FastSurfer labels and the chosen eigenfunction (1st or 2nd)
parcel = {

    ###  parcel_name : [ FastSurfer_labels, eigenfunction ]

        "Ventricles-no3d": ["4_5_31_43_44_63", 2],
        
        "Left-Lateral-Ventricle": ["4_5_31", 2],
        "Left-Inf-Lateral-Ventricle" : ["5", 2],
        "Right-Inf-Lateral-Ventricle" : ["44", 2],
        
        "3rd-Ventricle": ["14_24", 1],
        "4th-Ventricle": ["15", 1],
        "Left-VentralDC": ["28", 1],
    
        "Cerebellum": ["7_8_16_46_47", 1],
        "Brain-Stem": ["16", 1],
        
        "Left-Cerebellum-White-Matter": ["7", 1],
        "Left-Cerebellum-Cortex": ["8", 1],
        "Left-Thalamus-Proper": ["10", 1],
        "Left-Caudate": ["11", 1],
        "Left-Putamen": ["12", 1],
        "Left-Pallidum": ["13", 1],
        "Left-Hippocampus": ["17", 1],
        "Left-Amygdala": ["18", 1],
        "Left-Accumbens-area": ["26", 1],
        
        "ctx-lh-caudalanteriorcingulate": ["1002", 1],
        "ctx-lh-caudalmiddlefrontal": ["1003", 1], # difficult shape
        "ctx-lh-cuneus": ["1005", 1],
        "ctx-lh-entorhinal": ["1006", 1],
        "ctx-lh-fusiform": ["1007", 1],
        "ctx-lh-inferiorparietal": ["1008", 1], # difficult shape
        "ctx-lh-inferiortemporal": ["1009", 1],
        "ctx-lh-isthmuscingulate": ["1010", 1],
        'ctx-lh-lateraloccipital': ["1011", 1],
        'ctx-lh-lateralorbitofrontal': ["1012", 1],
        'ctx-lh-lingual': ["1013", 1],
        'ctx-lh-medialorbitofrontal': ["1014", 1],
        'ctx-lh-middletemporal': ["1015", 1],
        'ctx-lh-parahippocampal': ["1016", 1],
        'ctx-lh-paracentral': ["1017", 1],
        'ctx-lh-parsopercularis': ["1018", 1],
        'ctx-lh-parsorbitalis': ["1019", 1],
        'ctx-lh-parstriangularis': ["1020", 1],
        'ctx-lh-pericalcarine': ["1021", 1],
        'ctx-lh-postcentral': ["1022", 1],
        'ctx-lh-posteriorcingulate': ["1023", 1],
        'ctx-lh-precentral': ["1024", 1],
        'ctx-lh-precuneus': ["1025", 1],
        'ctx-lh-rostralanteriorcingulate': ["1026", 1],
        'ctx-lh-rostralmiddlefrontal': ["1027", 1],
        'ctx-lh-superiorfrontal': ["1028", 1],
        'ctx-lh-superiorparietal': ["1029", 1],
        'ctx-lh-superiortemporal': ["1030", 1],
        'ctx-lh-supramarginal': ["1031", 1], # difficult shape
        'ctx-lh-transversetemporal': ["1034", 1],
        'ctx-lh-insula': ["1035", 1],

        
        "Right-Lateral-Ventricle": ["43_44_63", 2],
        "Right-Cerebellum-White-Matter": ["46", 1],
        "Right-Cerebellum-Cortex": ["47", 1],
        "Right-Thalamus-Proper": ["49", 1],
        "Right-Caudate": ["50", 1],
        "Right-Putamen": ["51", 1],
        "Right-Pallidum": ["52", 1],
        "Right-Hippocampus": ["53", 1],
        "Right-Amygdala": ["54", 1],
        "Right-Accumbens-area": ["58", 1],
        "Right-VentralDC": ["60", 1],

        'ctx-rh-caudalanteriorcingulate': ["2002", 1],
        'ctx-rh-caudalmiddlefrontal': ["2003", 1],
        'ctx-rh-cuneus': ["2005", 1],
        'ctx-rh-entorhinal': ["2006", 1],
        'ctx-rh-fusiform': ["2007", 1],
        'ctx-rh-inferiorparietal': ["2008", 1],
        'ctx-rh-inferiortemporal': ["2009", 1],
        'ctx-rh-isthmuscingulate': ["2010", 1],
        'ctx-rh-lateraloccipital': ["2011", 1],
        'ctx-rh-lateralorbitofrontal': ["2012", 1],
        'ctx-rh-lingual': ["2013", 1],
        'ctx-rh-medialorbitofrontal': ["2014", 1],
        'ctx-rh-middletemporal': ["2015", 1],
        'ctx-rh-parahippocampal': ["2016", 1],
        'ctx-rh-paracentral': ["2017", 1],
        'ctx-rh-parsopercularis': ["2018", 1],
        'ctx-rh-parsorbitalis': ["2019", 1],
        'ctx-rh-parstriangularis': ["2020", 1],
        'ctx-rh-pericalcarine': ["2021", 1],
        'ctx-rh-postcentral': ["2022", 1],
        'ctx-rh-posteriorcingulate': ["2023", 1],
        'ctx-rh-precentral': ["2024", 1],
        'ctx-rh-precuneus': ["2025", 1],
        'ctx-rh-rostralanteriorcingulate': ["2026", 1],
        'ctx-rh-rostralmiddlefrontal': ["2027", 1],
        'ctx-rh-superiorfrontal': ["2028", 1],
        'ctx-rh-superiorparietal': ["2029", 1],
        'ctx-rh-superiortemporal': ["2030", 1],
        'ctx-rh-supramarginal': ["2031", 1],
        'ctx-rh-transversetemporal': ["2034", 1],
        'ctx-rh-insula': ["2035", 1]

        }

separation_avg = {} # we will populate this dictionary with the overall seperation that each shape can achieve (control/iNPH)
                # so that to sort the parcels afterward based on this method (apart from the explainability task which is the main focus).

separation_smooth_max = {}


# application of rings method >>> substitute with your own paths:
paths = [ "/media/bill/12EB-1B9D/UAS_dataset/UAS_controls_FS_SS/UAS_controls_FS_output/", 
          "/media/bill/12EB-1B9D/UAS_dataset/UAS_NPH_FS_SS/UAS_NPH_FS_output/",
          "/media/bill/12EB-1B9D/KS_dataset/KS_controls_FS_SS/KS_controls_FS_output/",
          "/media/bill/12EB-1B9D/KS_dataset/KS_NPH_FS_SS/KS_NPH_FS_output/"]

for par in parcel:
    errors = 0
    data = []

    for path in paths:
        subjects = os.listdir(path)
        for sub in subjects:
            if os.path.isdir(path+sub+"/brainprint/"):
                ev = pd.read_csv(path+sub+"/brainprint/eigenvectors/"+sub+".brainprint.evecs-"+par+".csv", index_col=0).to_numpy()
                seg = lapy.TriaMesh.read_vtk(path+sub+"/brainprint/surfaces/aseg.final."+parcel[par][0]+".vtk")
                try:
                    vf = diffgeo.tria_compute_geodesic_f(seg, ev[:, parcel[par][1] ])
                except RuntimeError as e:
                    errors += 1
                    continue
                z = np.cumsum( [0.01 for _ in range(98)] ) + 0.025/2
            
                vf0 = vf / np.max(vf)
                vf2 = np.ones_like(vf0)
            
                for j in range(len(z)):
            
                    for i in range(len(vf0)):
                        if vf0[i]>z[j] - 0.025/2 and vf0[i]<z[j]+0.025/2: # if vf0[i]>z[j] and vf0[i]<z[j]+0.025:
                            vf2[i] = j+1
                a=[]
                for k in range(1,99):
                    a.append( (vf2==k).sum() )
        
                if "_controls_" in path:
                    data.append( (a, 0 ) )
                elif "_NPH_" in path:
                    data.append( (a, 1) )
    
    # with open("/home/bill/Downloads/errors.txt", "a") as myfile:
    #     myfile.write(f"{par}: {errors}\n")

    for i in range(len(data)):
        if data[i][1] == 0:
            plt.plot(data[i][0], color="blue", alpha=0.2)
        else:
            plt.plot(data[i][0], color="red", alpha=0.2)
    plt.title(par)
    plt.show()
        
    sep=[]
    mw=[]
    for s in range(98):
        x=[]; y=[]
        for i in range(len(data)):
            if data[i][1] == 0:
                x.append( data[i][0][s] )
            else:
                y.append( data[i][0][s] )
            
        m = np.mean(x+y); sd = np.std(x+y)
        x = (x - m)/sd; y = (y - m)/sd
        sep.append( np.mean(x) - np.mean(y) )
        _, pv = stats.mannwhitneyu(x, y)
        mw.append(pv)
    separation_avg[par] = np.mean(np.abs(sep))
    separation_smooth_max[par] = np.max( np.abs(smoothed(sep, 3)) )
        
    plt.plot(sep)
    plt.title(par)
    plt.axhline(y=0, color='black', alpha=0.25)
    plt.savefig(f"{par}_sep.png")
    plt.close()

    plt.plot(mw)
    plt.title(par)
    plt.axhline(y=0.05, c="black", alpha=0.5, linestyle="--")
    plt.savefig(f"{par}_mw.png")
    plt.close()
        
    sub = "ASMRX13_impr_21_seg"
    path0 = "/media/bill/12EB-1B9D/UAS_dataset/UAS_NPH_FS_SS/UAS_NPH_FS_output/"
    ev = pd.read_csv(path0+sub+"/brainprint/eigenvectors/"+sub+".brainprint.evecs-"+par+".csv", index_col=0).to_numpy()
    seg = lapy.TriaMesh.read_vtk(path0+sub+"/brainprint/surfaces/aseg.final."+parcel[par][0]+".vtk")
    vf = diffgeo.tria_compute_geodesic_f(seg, ev[:,parcel[par][1]])
    vf = vf / np.max(vf)
    vf2 = np.zeros_like(vf)
        
    z = np.cumsum( [0.01 for i in range(98)] )
    for j in range(len(z)):
        
        for i in range(len(vf)):
            if vf[i]>z[j] and vf[i]<z[j]+0.025:
                vf2[i] =  sep[j]
        
    plot.plot_tria_mesh(seg, vf2, xrange=None, yrange=None, zrange=None, showcaxis=False, caxis=None)

print("finished.")
