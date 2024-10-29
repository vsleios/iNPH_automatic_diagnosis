### Implementation of:
# 1) development of models for radscale data + binary metrics (accuracy, sensitivity, specificity, F1-score)
# 2) measuring the triadic performance after training them on the whole dataset.


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# load the radscale data:
# -- substitute with your own paths
d = pd.read_excel("/media/bill/12EB-1B9D/Downloads/UAS_small_dataset_RadScale.xlsx")
uas_controls = pd.read_excel("/media/bill/12EB-1B9D/AgeSex/AgeSex_UAS_controls_final.xlsx")

# radscale data:
t=[]
for i in range(d.shape[0]):
    t.append( float( "_impr_" in d.iloc[i,0] ) )
    
    if d.iloc[i,2]=="F":
        d.iloc[i,2]=0
    if d.iloc[i,2]=="M":
        d.iloc[i,2]=1

d["iNPH"] = t
d = d.drop("NPH", axis=1)

# creating the labels:
labels = []
for i in range(d.shape[0]):
    if "_impr_" in d.iloc[i,0]:
        labels.append("iNPH")
    if "_contr_" in d.iloc[i,0]:
        s = float(d.iloc[i,0].split("_")[-2])
        diagnosis = list(uas_controls[uas_controls["ID"]==s]["Diagnosis"])[0]
        labels.append(diagnosis)


# helper functions:
#sensitivity specificity function + Confidence Interval + F1-score
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



# AGE, SEX, CA, EI:
print("\nAGE, SEX, CA, EI:")
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
data = d[["Age", "Sex", "CA", "EI", "iNPH"]]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

lb=[]; ub=[]; ib=[]
m1_tr=[]; m1_te_roc=[]; m1_te_def=[]
for train_index, test_index in kf.split(X=data, y=labels):
    x_train = data.iloc[train_index,:-1]; y_train = data.iloc[train_index,-1]
    x_test = data.iloc[test_index,:-1]; y_test = data.iloc[test_index,-1]
    x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
        
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
        
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)

    x_all=scaler.transform(x_all)
    x_all = pd.DataFrame(x_all)
        
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, C=0.2)
    lr.fit(x_train, y_train)
        
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    train_th = th[np.argmax(tpr - fpr)]
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_tr.append( [np.mean(q == y), a, b, c] )
        
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>train_th).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-roc: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_roc.append( [np.mean(q == y), a, b, c] )
        
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-0.5: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_def.append( [np.mean(q == y), a, b, c] )
    
    mat1_tr = np.array(m1_tr); mat1_te_roc = np.array(m1_te_roc); mat1_te_def = np.array(m1_te_def)

    ###
    y=list(y_all)
    p = lr.predict_proba(x_all)[:,1]
    
    cuts = np.cumsum( [0.01 for i in range(100)] )
    
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
    
    
    # plt.plot(l1)
    # plt.plot(l2)
    # plt.axhline(y=0.98, c="black", linewidth=1)
    # plt.show()
    
    c1, c2, c3 = cutoffs(l1, l2, p, 0.98)
    lb.append(c1); ub.append(c2); ib.append(c3)

print()
# print("with roc :", end=" ")
# for i in range(4):
#     mat=mat1_te_roc
#     print( np.mean(mat[:,i]), end=" " )

print()
print("acc, sen, spe, f1 :", end=" ")
for i in range(4):
    mat=mat1_te_def
    print( np.mean(mat[:,i]), end=" " )
print()
# print( np.mean(lb), np.mean(ub), np.mean(ib) )

#####
print("\n98% Triadic performance trained on the whole dataset :\n")
x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
    
scaler = StandardScaler()
scaler.fit(x_all)
x_all=scaler.transform(x_all)
x_all = pd.DataFrame(x_all)

lr = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, C=0.2)
lr.fit(x_all, y_all)

y=list(y_all)
p = lr.predict_proba(x_all)[:,1]
    
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
plt.savefig("./age_sex_ca_ei.png")
plt.close()

    
c1, c2, c3 = cutoffs(l1, l2, p, 0.98)


#AGE, SEX, CA, EI, TH:
print("\nAGE, SEX, CA, EI, TH:")
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
data = d[["Age", "Sex", "CA", "EI", "Temp", "iNPH"]]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

lb=[]; ub=[]; ib=[]
m1_tr=[]; m1_te_roc=[]; m1_te_def=[]
for train_index, test_index in kf.split(X=data, y=labels):
    x_train = data.iloc[train_index,:-1]; y_train = data.iloc[train_index,-1]
    x_test = data.iloc[test_index,:-1]; y_test = data.iloc[test_index,-1]
    x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
        
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
        
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)

    x_all=scaler.transform(x_all)
    x_all = pd.DataFrame(x_all)
        
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l2", C=1, max_iter=2000)
    lr.fit(x_train, y_train)
        
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    train_th = th[np.argmax(tpr - fpr)]
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_tr.append( [np.mean(q == y), a, b, c] )
        
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>train_th).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-roc: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_roc.append( [np.mean(q == y), a, b, c] )
        
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-0.5: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_def.append( [np.mean(q == y), a, b, c] )
    
    mat1_tr = np.array(m1_tr); mat1_te_roc = np.array(m1_te_roc); mat1_te_def = np.array(m1_te_def)

    ###
    y=list(y_all)
    p = lr.predict_proba(x_all)[:,1]
    
    cuts = np.cumsum( [0.01 for i in range(100)] )
    
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
    
    
    # plt.plot(l1)
    # plt.plot(l2)
    # plt.axhline(y=0.98, c="black", linewidth=1)
    # plt.show()
    
    c1, c2, c3 = cutoffs(l1, l2, p, 0.98)
    lb.append(c1); ub.append(c2); ib.append(c3)

print()
# print("with roc :", end=" ")
# for i in range(4):
#     mat=mat1_te_roc
#     print( np.mean(mat[:,i]), end=" " )

print()
print("acc, sen, spe, f1 :", end=" ")
for i in range(4):
    mat=mat1_te_def
    print( np.mean(mat[:,i]), end=" " )
print()
# print( np.mean(lb), np.mean(ub), np.mean(ib) )

#####
print("\n98% Triadic performance trained on the whole dataset :\n")
x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
    
scaler = StandardScaler()
scaler.fit(x_all)
x_all=scaler.transform(x_all)
x_all = pd.DataFrame(x_all)

lr = LogisticRegression(penalty="l2", max_iter=2000, C=1)
lr.fit(x_all, y_all)

y=list(y_all)
p = lr.predict_proba(x_all)[:,1]
    
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
plt.savefig("./age_sex_ca_ei_th.png")
plt.close()
    
c1, c2, c3 = cutoffs(l1, l2, p, 0.98)


#ALL RADSCALE + AGE, SEX:
print("\nALL RADSCALE + AGE, SEX:")
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
data = d[["Age", "Sex", "CA", "EI", "Temp", "vertex", "sylvi", "transport", "WMH", "iNPH"]]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

lb=[]; ub=[]; ib=[]
m1_tr=[]; m1_te_roc=[]; m1_te_def=[]
for train_index, test_index in kf.split(X=data, y=labels):
    x_train = data.iloc[train_index,:-1]; y_train = data.iloc[train_index,-1]
    x_test = data.iloc[test_index,:-1]; y_test = data.iloc[test_index,-1]
    x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
        
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
        
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)

    x_all=scaler.transform(x_all)
    x_all = pd.DataFrame(x_all)
        
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l2", C=0.5, max_iter=2000)
    lr.fit(x_train, y_train)
        
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    train_th = th[np.argmax(tpr - fpr)]
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_tr.append( [np.mean(q == y), a, b, c] )
        
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>train_th).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-roc: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_roc.append( [np.mean(q == y), a, b, c] )
        
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-0.5: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_def.append( [np.mean(q == y), a, b, c] )
    
    mat1_tr = np.array(m1_tr); mat1_te_roc = np.array(m1_te_roc); mat1_te_def = np.array(m1_te_def)

    ###
    y=list(y_all)
    p = lr.predict_proba(x_all)[:,1]
    
    cuts = np.cumsum( [0.01 for i in range(100)] )
    
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
    
    
    # plt.plot(l1)
    # plt.plot(l2)
    # plt.axhline(y=0.98, c="black", linewidth=1)
    # plt.show()
    
    c1, c2, c3 = cutoffs(l1, l2, p, 0.98)
    lb.append(c1); ub.append(c2); ib.append(c3)

print()
# print("with roc :", end=" ")
# for i in range(4):
#     mat=mat1_te_roc
#     print( np.mean(mat[:,i]), end=" " )

print()
print("acc, sen, spe, f1 :", end=" ")
for i in range(4):
    mat=mat1_te_def
    print( np.mean(mat[:,i]), end=" " )
print()
# print( np.mean(lb), np.mean(ub), np.mean(ib) )

#####
print("\n98% Triadic performance trained on the whole dataset :\n")
x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
    
scaler = StandardScaler()
scaler.fit(x_all)
x_all=scaler.transform(x_all)
x_all = pd.DataFrame(x_all)

lr = LogisticRegression(penalty="l2", C=0.5, max_iter=2000)
lr.fit(x_all, y_all)

y=list(y_all)
p = lr.predict_proba(x_all)[:,1]
    
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
plt.savefig("./all_rad.png")
plt.close()
    
c1, c2, c3 = cutoffs(l1, l2, p, 0.98)


# ALL RASDCALE WITHOUT AGE, SEX:
print("\nALL RASDCALE WITHOUT AGE, SEX:")
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
data = d[["CA", "EI", "Temp", "vertex", "sylvi", "transport", "WMH", "iNPH"]]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

lb=[]; ub=[]; ib=[]
m1_tr=[]; m1_te_roc=[]; m1_te_def=[]
for train_index, test_index in kf.split(X=data, y=labels):
    x_train = data.iloc[train_index,:-1]; y_train = data.iloc[train_index,-1]
    x_test = data.iloc[test_index,:-1]; y_test = data.iloc[test_index,-1]
    x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
        
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_train = pd.DataFrame(x_train)
        
    x_test=scaler.transform(x_test)
    x_test = pd.DataFrame(x_test)

    x_all=scaler.transform(x_all)
    x_all = pd.DataFrame(x_all)
        
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l2", C=0.5, max_iter=2000)
    lr.fit(x_train, y_train)
        
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    train_th = th[np.argmax(tpr - fpr)]
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_tr.append( [np.mean(q == y), a, b, c] )
        
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>train_th).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-roc: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_roc.append( [np.mean(q == y), a, b, c] )
        
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-0.5: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_def.append( [np.mean(q == y), a, b, c] )
    
    mat1_tr = np.array(m1_tr); mat1_te_roc = np.array(m1_te_roc); mat1_te_def = np.array(m1_te_def)

    ###
    y=list(y_all)
    p = lr.predict_proba(x_all)[:,1]
    
    cuts = np.cumsum( [0.01 for i in range(100)] )
    
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
    
    
    # plt.plot(l1)
    # plt.plot(l2)
    # plt.axhline(y=0.98, c="black", linewidth=1)
    # plt.show()
    
    c1, c2, c3 = cutoffs(l1, l2, p, 0.98)
    lb.append(c1); ub.append(c2); ib.append(c3)

print()
# print("with roc :", end=" ")
# for i in range(4):
#     mat=mat1_te_roc
#     print( np.mean(mat[:,i]), end=" " )

print()
print("acc, sen, spe, f1 :", end=" ")
for i in range(4):
    mat=mat1_te_def
    print( np.mean(mat[:,i]), end=" " )
print()
# print( np.mean(lb), np.mean(ub), np.mean(ib) )

#####
print("\n98% Triadic performance trained on the whole dataset :\n")
x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
    
scaler = StandardScaler()
scaler.fit(x_all)
x_all=scaler.transform(x_all)
x_all = pd.DataFrame(x_all)

lr = LogisticRegression(penalty="l2", C=0.5, max_iter=2000)
lr.fit(x_all, y_all)

y=list(y_all)
p = lr.predict_proba(x_all)[:,1]
    
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
plt.savefig("./all_rad_without_age_sex.png")
plt.close()
    
c1, c2, c3 = cutoffs(l1, l2, p, 0.98)


# RADSCALE UNIVARIATE:
print("\nRADSCALE UNIVARIATE:")
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
data = d[["RadScale Total", "iNPH"]]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

lb=[]; ub=[]; ib=[]
m1_tr=[]; m1_te_roc=[]; m1_te_def=[]
for train_index, test_index in kf.split(X=data, y=labels):
    x_train = data.iloc[train_index,:-1]; y_train = data.iloc[train_index,-1]
    x_test = data.iloc[test_index,:-1]; y_test = data.iloc[test_index,-1]
    x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
        
    from sklearn.preprocessing import StandardScaler
    
    # scaler = StandardScaler()
    # scaler.fit(x_train)
    # x_train=scaler.transform(x_train)
    # x_train = pd.DataFrame(x_train)
        
    # x_test=scaler.transform(x_test)
    # x_test = pd.DataFrame(x_test)

    # x_all=scaler.transform(x_all)
    # x_all = pd.DataFrame(x_all)
        
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=1, max_iter=2000)
    lr.fit(x_train, y_train)
        
    y=list(y_train)
    p = lr.predict_proba(x_train)[:,1]
    fpr, tpr, th = metrics.roc_curve(y, p, pos_label=1)
    train_th = th[np.argmax(tpr - fpr)]
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_tr.append( [np.mean(q == y), a, b, c] )
        
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>train_th).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-roc: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_roc.append( [np.mean(q == y), a, b, c] )
        
    q = (p>0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    # print("TEST-0.5: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    m1_te_def.append( [np.mean(q == y), a, b, c] )
    
    mat1_tr = np.array(m1_tr); mat1_te_roc = np.array(m1_te_roc); mat1_te_def = np.array(m1_te_def)

    ###
    y=list(y_all)
    p = lr.predict_proba(x_all)[:,1]
    
    cuts = np.cumsum( [0.01 for i in range(100)] )
    
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
    
    
    # plt.plot(l1)
    # plt.plot(l2)
    # plt.axhline(y=0.98, c="black", linewidth=1)
    # plt.show()
    
    c1, c2, c3 = cutoffs(l1, l2, p, 0.98)
    lb.append(c1); ub.append(c2); ib.append(c3)

x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]

print()
# print("with roc :", end=" ")
# for i in range(4):
#     mat=mat1_te_roc
#     print( np.mean(mat[:,i]), end=" " )

print()
print("acc, sen, spe, f1 :", end=" ")
for i in range(4):
    mat=mat1_te_def
    print( np.mean(mat[:,i]), end=" " )
print()
# print( np.mean(lb), np.mean(ub), np.mean(ib) )

#####
print("\n98% Triadic performance trained on the whole dataset :\n")
x_all = data.iloc[:,:-1]; y_all = data.iloc[:,-1]
    
# scaler = StandardScaler()
# scaler.fit(x_all)
# x_all=scaler.transform(x_all)
# x_all = pd.DataFrame(x_all)

lr = LogisticRegression(penalty="l1", solver="liblinear", C=1, max_iter=2000)
lr.fit(x_all, y_all)

y=list(y_all)
p = lr.predict_proba(x_all)[:,1]
    
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
plt.savefig("./rad_univariate.png")
plt.close() 
    
c1, c2, c3 = cutoffs(l1, l2, p, 0.98)

print("finished.")

