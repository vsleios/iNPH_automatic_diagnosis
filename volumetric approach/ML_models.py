from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


# loading the data:
data = pd.read_csv("./volumetric_data.csv", index_col=0)
labels = pd.read_csv("./volumetric_labels.csv", index_col=0)
labels = list(labels.iloc[:,0])


#############################sensitivity specificity function + Confidence Interval + F1-score
from statsmodels.stats.proportion import proportion_confint
def ss( pred, truth ):

    z = [ [pred[i], truth[i]] for i in range(len(truth)) ]
    n = len(pred)

    acc = np.mean(pred==truth)
    # acc_low, acc_up = proportion_confint(z.count([1,1])+z.count([0,0]), len(truth), alpha=0.05, method='wilson')
    acc_se = np.sqrt( acc*(1-acc)/n )
    (acc_low, acc_up) = (acc-1.96*acc_se, acc+1.96*acc_se)

    sens = z.count([1,1]) / ( z.count([1,1]) + z.count([0,1]) )
    # sens_low, sens_up = proportion_confint(z.count([1,1]), ( z.count([1,1]) + z.count([0,1]) ), alpha=0.05, method='wilson')
    sens_se = np.sqrt( sens*(1-sens)/n )
    (sens_low, sens_up) = (sens-1.96*sens_se, sens+1.96*sens_se)
    
    spec = z.count([0,0]) / ( z.count([0,0]) + z.count([1,0]) )
    # spec_low, spec_up = proportion_confint(z.count([0,0]), ( z.count([0,0]) + z.count([1,0]) ), alpha=0.05, method='wilson')
    spec_se = np.sqrt( spec*(1-spec)/n )
    (spec_low, spec_up) = (spec-1.96*spec_se, spec+1.96*spec_se)

    rec = sens
    pre = z.count([1,1]) / ( z.count([1,1]) + z.count([1,0]) )

    f1 = (2*pre*rec) / (pre + rec)

    return (round(acc_low,2), round(acc_up,2), sens, round(sens_low,2), round(sens_up,2), spec, round(spec_low,2), round(spec_up,2) , f1)




### 5-fold Cross validation, over ML models.

folds={}
for i in range(6):
    folds[i] = []

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
fold_acc_tr=[]; fold_sen_tr=[]; fold_spe_tr=[]
fold_acc_te=[]; fold_sen_te=[]; fold_spe_te=[]

for train_index, test_index in kf.split(X=data, y=labels):
    x_train = data.iloc[train_index,:-1]; y_train = data.iloc[train_index,-1]
    x_test = data.iloc[test_index,:-1]; y_test = data.iloc[test_index,-1]
    
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
    print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Log.Reg. AUC: {auc:.4f}")
    folds[0].append( np.array( [ np.mean(q == y), acc1,acc2, a,a1,a2, b,b1,b2, c ] ) )
    
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
    print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = lr2.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Lasso LR AUC: {auc:.4f}")
    folds[1].append( np.array( [ np.mean(q == y), acc1,acc2, a,a1,a2, b,b1,b2, c ] ) )
    
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
    print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = k.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"KNN AUC: {auc:.4f}")
    folds[2].append( np.array( [ np.mean(q == y), acc1,acc2, a,a1,a2, b,b1,b2, c ] ) )
    
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
    print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = dt.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Dec.Trees AUC: {auc:.4f}")
    folds[3].append( np.array( [ np.mean(q == y), acc1,acc2, a,a1,a2, b,b1,b2, c ] ) )
    
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
    print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = gb.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Gr. Boost. AUC: {auc:.4f}")
    folds[4].append( np.array( [ np.mean(q == y), acc1,acc2, a,a1,a2, b,b1,b2, c ] ) )
    
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
    print("TRAIN: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    
    y=list(y_test)
    p = bag.predict_proba(x_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y, p, pos_label=1)
    q = (p>=0.5).astype("int")
    acc1,acc2, a,a1,a2, b,b1,b2, c = ss( q , y )
    print("TEST: ", "Acc: ", np.mean(q == y), (acc1,acc2),"Sens: ", a,(a1,a2),"Spec: ", b,(b1,b2), "F1: ", c)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Bagging AUC: {auc:.4f}")
    folds[5].append( np.array( [ np.mean(q == y), acc1,acc2, a,a1,a2, b,b1,b2, c ] ) )
    
    plt.legend()
    # plt.title("ROC curves on Test set")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(f"./mlroc.png")
    plt.show()


# final results of the 5-fold CV:
print("\nAverage results over the 5 folds : \n")

print("Accuracy, Sensitivity, Specificity, F1-scores")
names = ["LR", "LR lasso", "KNN", "Dec.Trees", "Gr.Boosting", "Bagging"]
for i in range(6):
    f = list(sum( folds[i] ) / 5)
    g = [ str(round(e, 3)) for e in f ]
    s = names[i] + " & " + g[0]+" ("+g[1]+","+g[2]+") "+ "& " + g[3]+" ("+g[4]+","+g[5]+") "+ "& " + g[6]+" ("+g[7]+","+g[8]+") " + "& " + g[9] + " \\\\" 
    print(s)


# Odds ratios of the Logistic Regression model:
cols = data.columns[0:-1]
lr_imp = lr.coef_[0].copy()
# summarize feature importance
lr_dict = {}
for i,v in enumerate(lr_imp):
    lr_dict[ cols[i] ] = float(round(np.exp(v), 10))
    # lr_dict[ cols[i] ] = float(round(v, 10))
    
lr_dict = {k: v for k, v in sorted(lr_dict.items(), key=lambda item: -item[1])}

print(lr_dict)
