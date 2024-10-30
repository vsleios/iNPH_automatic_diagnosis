import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import metrics # for the roc_curve


# a random sample's brainprint.csv file (used to get the features' names) >>> substitute with your own path:
p = "/media/bill/12EB-1B9D/KS_dataset/KS_controls_FS_SS/KS_controls_FS_output/1_seg/brainprint/1_seg.brainprint.csv"
df = pd.read_csv(p, index_col=0).iloc[2:,:]

df = df.drop(["Ventricles", "Left-Right-Ventricles", "Left-Ventricle", "Right-Ventricle", "Left-Striatum", "Right-Striatum"], axis=1)
cols = list(df.columns)


# age sex data >>> substitute with your own paths:
agesex_uas_controls = pd.read_excel("/media/bill/12EB-1B9D/AgeSex/AgeSex_UAS_controls_final.xlsx")
agesex_uas_nph = pd.read_excel("/media/bill/12EB-1B9D/AgeSex/AgeSex_UAS_NPH_final.xlsx")
agesex_ks_controls = pd.read_excel("/media/bill/12EB-1B9D/AgeSex/AgeSex_KS_controls_final.xlsx")
agesex_ks_nph = pd.read_csv("/media/bill/12EB-1B9D/AgeSex/AgeSex_KS_NPH_final.csv", delimiter=";")

gender = {"F": 0, "M": 1}


# computation of silhouette scores
struct_ev = {}
for col in cols:
    struct_ev[col] = []

paths = [ "/media/bill/12EB-1B9D/UAS_dataset/UAS_controls_FS_SS/UAS_controls_FS_output", 
          "/media/bill/12EB-1B9D/UAS_dataset/UAS_NPH_FS_SS/UAS_NPH_FS_output",
          "/media/bill/12EB-1B9D/KS_dataset/KS_controls_FS_SS/KS_controls_FS_output",
          "/media/bill/12EB-1B9D/KS_dataset/KS_NPH_FS_SS/KS_NPH_FS_output"]

agesex_list = [agesex_uas_controls, agesex_uas_nph, agesex_ks_controls, agesex_ks_nph]

labels = []
flag_p_plus=False

for i in range(4):
    path = paths[i]

    ###
    files = os.listdir(path)
    for subj in files:
        if os.path.isdir(path+f"/{subj}/brainprint/"):
            if "ASMRX12" not in subj:
                s = float(subj.split("_")[-2])
            if "ASMRX12" in subj:
                s = subj[0:-4]
    
            if "_NPH_" in path:
                labels.append("iNPH")
            else:
                t = agesex_list[i]
                diagnosis = list(t[t["ID"]==s]["Diagnosis"])[0]
                if diagnosis == "CBS or MSA":
                    diagnosis = "MSA-P"
                if diagnosis == "P+":
                    if flag_p_plus==False:
                        diagnosis = "MSA-P"
                        flag_p_plus = True
                    elif flag_p_plus == True:
                        diagnosis = "PSP"
                labels.append(diagnosis)
            
            extension = f"/{subj}/brainprint/{subj}.brainprint.csv"
            
            d = pd.read_csv(path+extension, index_col=0).iloc[2:,:]
            for col in cols:
                row = list(d[col])
                row.append(float("_NPH_" in path))
                struct_ev[col].append(row)

for col in cols:
    struct_ev[col] = pd.DataFrame(struct_ev[col])


# renamings:
struct_ev["Infratentorial-Brain"] = struct_ev.pop("Cerebellum")
struct_ev["Lateral-Ventricles"] = struct_ev.pop("Ventricles-no3d")

scores = {}
for col in struct_ev.keys():
    scores[col] = abs(silhouette_score( struct_ev[col].iloc[:,0:-1], list(struct_ev[col].iloc[:,-1])) )

scores = {k: round(v,3) for k, v in sorted(scores.items(), key=lambda item: -item[1])}

print("Silhouette scores of the structures : \n")
print(scores)


### visualization examples of clusterings:
for ind, i in enumerate([11, 6, 1, 2]):
    d = struct_ev[cols[i]].iloc[:,0:-1]
    lbls = struct_ev[cols[i]].iloc[:,-1]
    pca = PCA()
    pca_comp = pca.fit_transform(d) # principal components
    comp = pd.DataFrame(pca_comp)
    comp.iloc[:,0]
    sc = plt.scatter(x=comp.iloc[:,0], y=comp.iloc[:,1], c=lbls)
    plt.title(str(cols[i])+" "+str(round( scores[cols[i]], 3) ))
    plt.legend(handles=sc.legend_elements()[0], labels=["contr", "iNPH"])
    plt.savefig(f"./clust{ind}.png")
    plt.close()


# ML models: iNPH prediction using BrainPrint (i.e. eigenvalues) data

# creation of the dataframe:
E = []
for i in range(376):
    row=[]
    for col in struct_ev.keys():
        row = row + list(struct_ev[col].iloc[i,0:-1])
    row.append(struct_ev[cols[0]].iloc[i,-1])
    E.append(row)
ev = pd.DataFrame(E)


# helper functions :

###sensitivity specificity function + Confidence Interval + F1-score
from statsmodels.stats.proportion import proportion_confint
def ss( pred, truth ):

    z = [ [pred[i], truth[i]] for i in range(len(truth)) ]

    acc_low, acc_up = proportion_confint(z.count([1,1])+z.count([0,0]), len(truth), alpha=0.05, method='wilson')

    sens = z.count([1,1]) / ( z.count([1,1]) + z.count([0,1]) )
    sens_low, sens_up = proportion_confint(z.count([1,1]), ( z.count([1,1]) + z.count([0,1]) ), alpha=0.05, method='wilson')
    
    spec = z.count([0,0]) / ( z.count([0,0]) + z.count([1,0]) )
    spec_low, spec_up = proportion_confint(z.count([0,0]), ( z.count([0,0]) + z.count([1,0]) ), alpha=0.05, method='wilson')

    rec = sens
    pre = z.count([1,1]) / ( z.count([1,1]) + z.count([1,0]) )

    f1 = (2*pre*rec) / (pre + rec)

    return (round(acc_low,2), round(acc_up,2), sens, round(sens_low,2), round(sens_up,2), spec, round(spec_low,2), round(spec_up,2) , f1)

### computes the Confidence Interval
def make_ci(acc, sens, spec, n):
    
    acc_se = np.sqrt( acc*(1-acc)/n )
    (acc_low, acc_up) = (acc-1.96*acc_se, acc+1.96*acc_se)

    sens_se = np.sqrt( sens*(1-sens)/n )
    (sens_low, sens_up) = (sens-1.96*sens_se, sens+1.96*sens_se)

    spec_se = np.sqrt( spec*(1-spec)/n )
    (spec_low, spec_up) = (spec-1.96*spec_se, spec+1.96*spec_se)

    acc=round(acc,3); sens=round(sens,3); spec=round(spec,3)
    (acc_low, acc_up)=(round(acc_low,3), round(acc_up,3) )
    (sens_low, sens_up)=( round(sens_low,3), round(sens_up,3) )
    (spec_low, spec_up)=( round(spec_low,3), round(spec_up,3) )

    if acc_up>1:
        acc_up = 1
    if sens_up>1:
        sens_up = 1
    if spec_up>1:
        spec_up = 1
    
    x=( f"{acc} ({acc_low},{acc_up}) & {sens} ({sens_low},{sens_up}) & {spec} ({spec_low},{spec_up})" )
    return(x)



print("\nCross Validation folds :")
m1=[]; m2=[]; m3=[]; m4=[]; m5=[]; m6=[]
for i in range(5):
    print(f"\nFold {i}")
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(ev.drop([4550], axis=1), ev[4550], stratify=labels, random_state=13, test_size=0.2)
    
    
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    ########### Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print( "lr :")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    train_th = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f"Log.Reg. AUC: {auc:.4f}")
    m1.append( [np.mean(q == y), a, b, c] )
    
    ############################ Logistic Regression Lasso Regularization
    from sklearn.linear_model import LogisticRegression
    lr2 = LogisticRegression(penalty='l1', solver='liblinear')
    lr2.fit(x_train, y_train)
    print( "lr lasso :")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr2.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    train_th = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr2.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f"Lasso LR AUC: {auc:.4f}")
    m2.append( [np.mean(q == y), a, b, c] )
    
    ##################### KNN
    import sklearn.neighbors as knn
    k = knn.KNeighborsClassifier(n_neighbors=10)
    k.fit(x_train,y_train)
    print("knn :")#, np.mean(k.predict(x_train) == y_train), np.mean(k.predict(x_test) == y_test))
    
    y=list(y_train)
    p = k.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    train_th = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = k.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f"KNN AUC: {auc:.4f}")
    m3.append( [np.mean(q == y), a, b, c] )
    
    ########################### DECISION TREES
    import sklearn.tree as tree
    dt = tree.DecisionTreeClassifier(max_depth=5)
    dt.fit(x_train,y_train)
    print("dt : ")#, np.mean( (dt.predict(x_train)>0.5).astype(int) == y_train), np.mean( (dt.predict(x_test)>0.5).astype(int) == y_test) )
    
    y=list(y_train)
    p = dt.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    train_th = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = dt.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f"Dec.Trees AUC: {auc:.4f}")
    m4.append( [np.mean(q == y), a, b, c] )
    
    ##### Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier()
    gb.fit(x_train,y_train)
    print("gb : ")
    
    y=list(y_train)
    p = gb.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    train_th = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = gb.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f"Gr. Boost. AUC: {auc:.4f}")
    m5.append( [np.mean(q == y), a, b, c] )
    
    #### Bagging
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import SVR
    bag = BaggingClassifier(n_estimators=10)
    bag.fit(x_train,y_train)
    print("bag : ")
    
    y=list(y_train)
    p = bag.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    train_th = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = bag.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f"Bagging AUC: {auc:.4f}")
    m6.append( [np.mean(q == y), a, b, c] )
    
    # plt.legend()
    # # plt.title("ROC curves on Test set")
    # plt.xlabel("False positive rate")
    # plt.ylabel("True positive rate")
    # plt.savefig(f"./bproc.png")
    # plt.close()

print("\nCross Validation average: \n")
names=["LR", "LR lasso", "KNN", "Dec.Trees", "Gr.Boosting", "Bagging"]
mats = [m1, m2, m3, m4, m5, m6]
for i in range(len(names)):
    mat = np.array(mats[i])
    print( names[i] + " & " + make_ci(np.mean(mat[:,0]), np.mean(mat[:,1]), np.mean(mat[:,2]), 76) + f" & {round(np.mean(mat[:,3]), 3)} \\\\" )

print("\nfinished.")
