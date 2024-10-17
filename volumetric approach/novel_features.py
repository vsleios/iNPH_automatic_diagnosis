import numpy as np
import pandas as pd
from sklearn import metrics


# loading the data:
data = pd.read_csv("./volumetric_data.csv", index_col=0)
labels = pd.read_csv("./volumetric_labels.csv", index_col=0)
labels = list(labels.iloc[:,0])


### sensitivity specificity function + Confidence Interval + F1-score
from statsmodels.stats.proportion import proportion_confint
def ss( pred, truth ):

    z = [ [pred[i], truth[i]] for i in range(len(truth)) ]
    n = len(pred)

    acc = np.mean(pred==truth)
    acc_se = np.sqrt( acc*(1-acc)/n )
    (acc_low, acc_up) = (acc-1.96*acc_se, acc+1.96*acc_se)

    sens = z.count([1,1]) / ( z.count([1,1]) + z.count([0,1]) )
    sens_se = np.sqrt( sens*(1-sens)/n )
    (sens_low, sens_up) = (sens-1.96*sens_se, sens+1.96*sens_se)
    
    spec = z.count([0,0]) / ( z.count([0,0]) + z.count([1,0]) )
    spec_se = np.sqrt( spec*(1-spec)/n )
    (spec_low, spec_up) = (spec-1.96*spec_se, spec+1.96*spec_se)

    rec = sens
    pre = z.count([1,1]) / ( z.count([1,1]) + z.count([1,0]) )

    f1 = (2*pre*rec) / (pre + rec)

    return (round(acc_low,2), round(acc_up,2), sens, round(sens_low,2), round(sens_up,2), spec, round(spec_low,2), round(spec_up,2) , f1)



### Novel features ML models - 5-fold Cross Validation.
### The novel features are computed accordingly, based on the original columns of the dataset (volumetric_data.csv)

from sklearn.model_selection import StratifiedKFold

met1 = [ [] for i in range(7) ] # 0,1,2 : train   3,4,5: test + 6 test f1-score.
met2 = [ [] for i in range(7) ]
met3 = [ [] for i in range(7) ]
met4 = [ [] for i in range(7) ]
met5 = [ [] for i in range(7) ]
met6 = [ [] for i in range(7) ]
met7 = [ [] for i in range(7) ]
met8 = [ [] for i in range(7) ]
met9 = [ [] for i in range(7) ]
met10 = [ [] for i in range(7) ]
met11 = [ [] for i in range(7) ]
met12 = [ [] for i in range(7) ]


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
for train_index, test_index in kf.split(X=data, y=labels):

    train = data.iloc[train_index,:]; test = data.iloc[test_index,:]

    
    ### TRAIN :
    df = train.copy()
    
    ### choroid plexus
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( df.iloc[i,31] + df.iloc[i,17] )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d0 = pd.DataFrame(E)
    
    ### 3D Evan's Index
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( df.iloc[i,1] + df.iloc[i,2] + df.iloc[i,19] + df.iloc[i,20] )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d1 = pd.DataFrame(E)
    
    ### 3D Evan's Index (with choroid plexus)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( df.iloc[i,1] + df.iloc[i,2] + df.iloc[i,19] + df.iloc[i,20] + df.iloc[i,17] + df.iloc[i,31] )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d1_cp = pd.DataFrame(E)
    
    
    ### VV1 / parenchyma (without choroid plexus)
    E = []
    for i in range(df.shape[0]):
        row=[]
        # 31, 17: Left and Right choroid plexus
        row.append( np.sum(df.iloc[i,[1,2,19,20]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d2 = pd.DataFrame(E)
    
    ### VV1 / parenchyma (with choroid plexus)
    E = []
    for i in range(df.shape[0]):
        row=[]
        # 31, 17: Left and Right choroid plexus
        row.append( np.sum(df.iloc[i,[1,2,19,20,17,31]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d2_cp = pd.DataFrame(E)
    
    
    ### VV2/parenchyma (without cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d3 = pd.DataFrame(E)
    
    ### VV2/parenchyma (with cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10, 17,31]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d3_cp = pd.DataFrame(E)
    
    ### VV1 / parenchyma cerebral (without cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31, 17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d4 = pd.DataFrame(E)
    
    ### VV1 / parenchyma cerebral (with cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20, 31, 17]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31, 17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d4_cp = pd.DataFrame(E)
    
    ### VV2 / parenchyma cerebral
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d5 = pd.DataFrame(E)
    
    ### VV2 / parenchyma cerebral
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10,31,17]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d5_cp = pd.DataFrame(E)
    
    ### L+R inferior Vents / L+R Hippocampus # (2+20) / (12+27)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[2, 20]]) / np.sum(df.iloc[i,[12, 27]]) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d6 = pd.DataFrame(E)
    
    #########################################################################

    ### TEST
    df = test.copy()
    
    ### choroid plexus
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( df.iloc[i,31] + df.iloc[i,17] )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d0_test= pd.DataFrame(E)
    
    ### 3D Evan's Index
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( df.iloc[i,1] + df.iloc[i,2] + df.iloc[i,19] + df.iloc[i,20] )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d1_test = pd.DataFrame(E)
    
    ### 3D Evan's Index (with choroid plexus)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( df.iloc[i,1] + df.iloc[i,2] + df.iloc[i,19] + df.iloc[i,20] + df.iloc[i,17] + df.iloc[i,31] )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d1_cp_test = pd.DataFrame(E)
    
    
    ### VV1 / parenchyma (without choroid plexus)
    E = []
    for i in range(df.shape[0]):
        row=[]
        # 31, 17: Left and Right choroid plexus
        row.append( np.sum(df.iloc[i,[1,2,19,20]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d2_test = pd.DataFrame(E)
    
    ### VV1 / parenchyma (with choroid plexus)
    E = []
    for i in range(df.shape[0]):
        row=[]
        # 31, 17: Left and Right choroid plexus
        row.append( np.sum(df.iloc[i,[1,2,19,20,17,31]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d2_cp_test = pd.DataFrame(E)
    
    
    ### VV2/parenchyma (without cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d3_test = pd.DataFrame(E)
    
    ### VV2/parenchyma (with cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10, 17,31]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d3_cp_test = pd.DataFrame(E)
    
    ### VV1 / parenchyma cerebral (without cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31, 17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d4_test = pd.DataFrame(E)
    
    ### VV1 / parenchyma cerebral (with cp)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20, 31, 17]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31, 17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d4_cp_test = pd.DataFrame(E)
    
    ### VV2 / parenchyma cerebral
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d5_test = pd.DataFrame(E)
    
    ### VV2 / parenchyma cerebral
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[1,2,19,20,9,10,31,17]]) / ( np.sum(df.iloc[i,:-3]) - np.sum(df.iloc[i,[1,2,19,20,9,10,32,14,11,3,4,21,22, 31,17]])) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d5_cp_test = pd.DataFrame(E)
    
    ### L+R inferior Vents / L+R Hippocampus # (2+20) / (12+27)
    E = []
    for i in range(df.shape[0]):
        row=[]
        row.append( np.sum(df.iloc[i,[2, 20]]) / np.sum(df.iloc[i,[12, 27]]) )
        row.append( df.iloc[i,-1] )
        E.append(row)
    d6_test = pd.DataFrame(E)
    
    #########################
    ######## models #########
    
    tr, te = d0, d0_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nChoroid Plexus  :")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met1[0].append(np.mean(q == y)); met1[1].append(a); met1[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    met1[3].append(np.mean(q == y)); met1[4].append(a); met1[5].append(b); met1[6].append(c)
    
    ##########################
    
    tr, te = d1, d1_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\n3D Evan'sIndex  :")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met2[0].append(np.mean(q == y)); met2[1].append(a); met2[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met2[3].append(np.mean(q == y)); met2[4].append(a); met2[5].append(b); met2[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    ###################
    
    tr, te = d1_cp, d1_cp_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\n3D Evan'sIndex cp :")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met3[0].append(np.mean(q == y)); met3[1].append(a); met3[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met3[3].append(np.mean(q == y)); met3[4].append(a); met3[5].append(b); met3[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    
    #################
    
    tr, te = d2, d2_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV1/parenchyma:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met4[0].append(np.mean(q == y)); met4[1].append(a); met4[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met4[3].append(np.mean(q == y)); met4[4].append(a); met4[5].append(b); met4[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    
    #################
    
    tr, te = d2_cp, d2_cp_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV1/parenchyma cp:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met5[0].append(np.mean(q == y)); met5[1].append(a); met5[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met5[3].append(np.mean(q == y)); met5[4].append(a); met5[5].append(b); met5[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    
    #################
    
    tr, te = d3, d3_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV2/parenchyma:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met6[0].append(np.mean(q == y)); met6[1].append(a); met6[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met6[3].append(np.mean(q == y)); met6[4].append(a); met6[5].append(b); met6[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    
    ##################
    
    tr, te = d3_cp, d3_cp_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV2/parenchyma cp:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met7[0].append(np.mean(q == y)); met7[1].append(a); met7[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met7[3].append(np.mean(q == y)); met7[4].append(a); met7[5].append(b); met7[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    
    #################
    
    tr, te = d4, d4_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV1/parenchyma cerebral:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met8[0].append(np.mean(q == y)); met8[1].append(a); met8[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met8[3].append(np.mean(q == y)); met8[4].append(a); met8[5].append(b); met8[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    
    #################
    
    tr, te = d4_cp, d4_cp_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV1/parenchyma cerebral cp:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met9[0].append(np.mean(q == y)); met9[1].append(a); met9[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met9[3].append(np.mean(q == y)); met9[4].append(a); met9[5].append(b); met9[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    #################
    
    tr, te = d5, d5_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV2/parenchyma cerebral:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met10[0].append(np.mean(q == y)); met10[1].append(a); met10[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met10[3].append(np.mean(q == y)); met10[4].append(a); met10[5].append(b); met10[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    ##################
    
    
    tr, te = d5_cp, d5_cp_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nVV2/parenchyma cerebral cp:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met11[0].append(np.mean(q == y)); met11[1].append(a); met11[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met11[3].append(np.mean(q == y)); met11[4].append(a); met11[5].append(b); met11[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    
    #################
    
    tr, te = d6, d6_test
    
    x_train = np.array(tr.iloc[:,0]).reshape(-1, 1); y_train = tr.iloc[:,1]
    x_test = np.array(te.iloc[:,0]).reshape(-1, 1); y_test = te.iloc[:,1]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
    
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print( "\nHip+ParahipV/THV:")#, np.mean( lr.predict(x_train) == y_train ) )
    
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=th[np.argmax(tpr - fpr)]).astype("int")
    th_train = th[np.argmax(tpr - fpr)]
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met12[0].append(np.mean(q == y)); met12[1].append(a); met12[2].append(b)
    # print("TRAIN: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    met12[3].append(np.mean(q == y)); met12[4].append(a); met12[5].append(b); met12[6].append(c)
    # print("TEST: ", "Acc: ", round(np.mean(q == y),3), (acc1,acc2),"Sens: ", round(a,3),(a1,a2),"Spec: ", round(b,3),(b1,b2), "F1: ", c)
    

### We average all 4 metrics (accuracy, sensitivity, specificity, f1-score) over the 5-folds.
### and print the final results in the Train and Test set.

print("\nChoroid Plexus")
print(f"TRAIN: {np.mean(met1[0])}, {np.mean(met1[1])}, {np.mean(met1[2])}")
print(f"TEST: {np.mean(met1[3])}, {np.mean(met1[4])}, {np.mean(met1[5])}, {np.mean(met1[6])}")
print("\n3D EI")
print(f"TRAIN: {np.mean(met2[0])}, {np.mean(met2[1])}, {np.mean(met2[2])}")
print(f"TEST: {np.mean(met2[3])}, {np.mean(met2[4])}, {np.mean(met2[5])}, {np.mean(met2[6])}")
print("\n3D EI cp")
print(f"TRAIN: {np.mean(met3[0])}, {np.mean(met3[1])}, {np.mean(met3[2])}")
print(f"TEST: {np.mean(met3[3])}, {np.mean(met3[4])}, {np.mean(met3[5])}, {np.mean(met3[6])}")
print("\nVV1/parenchyma")
print(f"TRAIN: {np.mean(met4[0])}, {np.mean(met4[1])}, {np.mean(met4[2])}")
print(f"TEST: {np.mean(met4[3])}, {np.mean(met4[4])}, {np.mean(met4[5])}, {np.mean(met4[6])}")
print("\nVV1/parenchyma cp")
print(f"TRAIN: {np.mean(met5[0])}, {np.mean(met5[1])}, {np.mean(met5[2])}")
print(f"TEST: {np.mean(met5[3])}, {np.mean(met5[4])}, {np.mean(met5[5])}, {np.mean(met5[6])}")
print("\nVV2/parenchyma")
print(f"TRAIN: {np.mean(met6[0])}, {np.mean(met6[1])}, {np.mean(met6[2])}")
print(f"TEST: {np.mean(met6[3])}, {np.mean(met6[4])}, {np.mean(met6[5])}, {np.mean(met6[6])}")
print("\nVV2/parenchyma cp")
print(f"TRAIN: {np.mean(met7[0])}, {np.mean(met7[1])}, {np.mean(met7[2])}")
print(f"TEST: {np.mean(met7[3])}, {np.mean(met7[4])}, {np.mean(met7[5])}, {np.mean(met7[6])}")
print("\nVV1/parenchyma cerebral")
print(f"TRAIN: {np.mean(met8[0])}, {np.mean(met8[1])}, {np.mean(met8[2])}")
print(f"TEST: {np.mean(met8[3])}, {np.mean(met8[4])}, {np.mean(met8[5])}, {np.mean(met8[6])}")
print("\nVV1/parenchyma cerebral cp")
print(f"TRAIN: {np.mean(met9[0])}, {np.mean(met9[1])}, {np.mean(met9[2])}")
print(f"TEST: {np.mean(met9[3])}, {np.mean(met9[4])}, {np.mean(met9[5])}, {np.mean(met9[6])}")
print("\nVV2/parenchyma cerebral")
print(f"TRAIN: {np.mean(met10[0])}, {np.mean(met10[1])}, {np.mean(met10[2])}")
print(f"TEST: {np.mean(met10[3])}, {np.mean(met10[4])}, {np.mean(met10[5])}, {np.mean(met10[6])}")
print("\nVV2/parenchyma cerebral cp")
print(f"TRAIN: {np.mean(met11[0])}, {np.mean(met11[1])}, {np.mean(met11[2])}")
print(f"TEST: {np.mean(met11[3])}, {np.mean(met11[4])}, {np.mean(met11[5])}, {np.mean(met11[6])}")
print("\nL+R InfVents / L+R Hippocampus")
print(f"TRAIN: {np.mean(met12[0])}, {np.mean(met12[1])}, {np.mean(met12[2])}")
print(f"TEST: {np.mean(met12[3])}, {np.mean(met12[4])}, {np.mean(met12[5])}, {np.mean(met12[6])}")

