import numpy as np
import pandas as pd

#/media/bill/12EB-1B9D/Downloads/UAS_small dataset_RadScale.xlsx

d = pd.read_excel("./radscale_data.xlsx")

t=[]
for i in range(d.shape[0]):
    t.append( float( "_impr_" in d.iloc[i,0] ) )
    
    if d.iloc[i,2]=="F":
        d.iloc[i,2]=0
    if d.iloc[i,2]=="M":
        d.iloc[i,2]=1

d = d.drop(["ID", "NPH"], axis=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Rad1(a): # a: array or of the features
    a = np.array( [a[0:4]] )
    #scaler parameters:
    mean = np.array([73.18014184, 0.53191489, 85.09929078, 0.32037137])
    scale = np.array([ 6.41815046, 0.4989804, 23.65661707, 0.04866711])

    #scaling:
    b = (a - mean) / scale

    #lr parameters:
    coef = np.array([ 0, 0, -2.07016193, 0.55619858])
    interc = 0

    #lr prediction:
    p = sigmoid( np.dot(b, coef) + interc )
    
    return( p )

def Rad2(a): # a: array or of the features
    a = np.array( [a[0:5]] )
    #scaler parameters:
    mean = np.array([73.18014184,  0.53191489, 85.09929078,  0.32037137,  5.02553191])
    scale = np.array([ 6.41815046,  0.4989804 , 23.65661707,  0.04866711,  1.99404803])

    #scaling:
    b = (a - mean) / scale

    #lr parameters:
    coef = np.array([-0.4833648 , -0.32802881, -2.32989875,  0.73007967,  0.73829806])
    interc = -0.34999355

    #lr prediction:
    p = sigmoid( np.dot(b, coef) + interc )
    
    return( p )

def Rad3(a): # a: array or of the features
    a = np.array( [a[0:-1]] )
    #scaler parameters:
    mean = np.array([7.31801418e+01, 5.31914894e-01, 8.50992908e+01, 3.20371374e-01,
                     5.02553191e+00, 6.17021277e-01, 3.97163121e-01, 7.80141844e-02, 6.59574468e-01])
    scale = np.array([ 6.41815046,  0.4989804 , 23.65661707,  0.04866711,  1.99404803,
                     0.85601397,  0.48931031,  0.26819391,  0.7794323 ])

    #scaling:
    b = (a - mean) / scale

    #lr parameters:
    coef = np.array([-0.5054203, 0.09092197, -1.19484571, 0.72121796, 0.72976914,
                      0.15505336, 1.38426581, 0.2445758, 0.32306939])
    interc = -0.20382964

    #lr prediction:
    p = sigmoid( np.dot(b, coef) + interc )
    
    return( p )

def Rad4(a): # a: array or of the features
    a = a[-1]
    #No scaling.

    #lr parameters:
    coef = 1.01751327
    interc = -5.34971293

    #lr prediction:
    p = sigmoid( a * coef + interc )
    
    return( p )



print("\nPredictions of 4 Radscale models :")
print("4 probability values from each model in order: prob_model1, ..., prob_model4.\n")

print("model1: Age Sex CA EI")
print("model2: Age Sex CA EI TH")
print("model3: All radscale features")
print("model4: Univariate classic radscale\n") 

print("output in rad_output.csv")

E=[]
for i in range(d.shape[0]):
    row = [ np.float64(e) for e in list(d.iloc[i,:]) ]
    E.append( [ Rad1(row)[0], Rad2(row)[0], Rad3(row)[0], Rad4(row) ] )

out = pd.DataFrame(E)
out.to_csv("./rad_output.csv", sep='\t')

print("finished\n")

