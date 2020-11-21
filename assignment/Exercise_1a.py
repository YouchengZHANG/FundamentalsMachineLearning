
## Exercise 1a

import numpy as np
from numpy.core.fromnumeric import mean, std
import matplotlib.pyplot as plt # plotting
from statistics import *

## 1 Monte-Carlo Simulation
## 1.1 Data Creation and Visualization (7 points)

# Generate CDF
# F(x)[p(X = x|Y = 0)] = 2x - x^2
# F(x)[p(X = x|Y = 1)] = x^2

# Find inverse CDF
# F'(x)[p(X = x|Y = 0)] = 1 - sqrt(1-y)   * 0<x<1
# F'(x)[p(X = x|Y = 1)] = sqrt(y)

def create_data(N):

    # Generate Y 
    Y = np.random.choice([0,1], size = N, p = [0.5,0.5])   # shape:(N,)

    # Initialize X 
    X = np.array([])    # shape:(,)

    for y in Y:

        # Generate uniformly distributed sample for inverse CDF method
        u = np.random.uniform(0, 1)

        # Generate X using inverse CDF method
        if y == 0 : 
            X = np.append( X, 1 - np.sqrt(1-u) )
        else :
            X = np.append( X, np.sqrt(u) )
    
    return X,Y

# Return X-values, Y-labels for N=10000 instances 
N = 10000
X , Y = create_data(N)

# Check the distribution with matplotlib (X)
plt.hist(X,density=True,color = "skyblue")

# Check the distribution with matplotlib (likelihood)
plt.hist(X[np.where(Y == 0)], color = "deepskyblue")
plt.hist(X[np.where(Y == 1)], color = "deepskyblue")


## 1.2 Classication by Thresholding (5 points)

def RuleClassifier(x, xt, rule):
    y = []
    if rule == "A":
        for xi in x:
            if xi < xt:
                y.append(0)
            else:
                y.append(1)
            
    elif rule == "B":
        for xi in x:
            if xi < xt:
                y.append(1)
            else:
                y.append(0)

    elif rule == "C":
        for xi in x:
            yn = np.random.choice(2, size = 1).tolist()[0]
            y.append(yn)

    elif rule == "D":
        for xi in x:
            y.append(1)

    return y
        
def calError(xt) : 
    probErrorA = 1/4 + (xt - 1/2) ** 2
    probErrorB = 1 - probErrorA
    return print("Rule A --> Error =",probErrorA, ";", "Rule B --> Error =", probErrorB)

def calRealError(xt, M, n, model, rule):

    MeanA, DevA, MeanB, DevB  = [], [], [], []

    for m in M:
        
        errorA, errorB = [], []
        for _ in range(n):
            X , Y = create_data(m)
            yA = model(X, xt, rule=rule[0])
            errorA.append( (m - np.sum(yA == Y)) / m )
            yB = model(X, xt, rule=rule[1])
            errorB.append( (m - np.sum(yB == Y)) / m )

        meanErrorA, devErrorA, meanErrorB, devErrorB = np.mean((errorA)),np.std(errorA),np.mean(errorB),np.std(errorB)

        MeanA.append(meanErrorA)
        DevA.append(devErrorA)
        MeanB.append(meanErrorB)
        DevB.append(devErrorB)

        print("Rule",rule[0],"( xt = ",xt,")--> M =",m," : mean =",round(meanErrorA,2)," ; ","sd =",round(devErrorA,3))
        print("Rule",rule[1],"( xt = ",xt,")--> M =",m," : mean =",round(meanErrorB,2)," ; ","sd =",round(devErrorB,3))

    return MeanA, DevA, MeanB, DevB


# Calculate the 'true' error rate using given rate formula
calError(xt=0.2)
calError(xt=0.5)
calError(xt=0.6)

# Calculate the 'real' error rate by repeating 10 test datasets of the same size M = [10,100,1000,1000]
M = [10,100,1000,10000]
# xt = 0.2
MeanA1, DevA1, MeanB1, DevB1 = calRealError(xt=0.2, M=M, n=10, model=RuleClassifier,rule=["A","B"])

# xt = 0.5
MeanA2, DevA2, MeanB2, DevB2 = calRealError(xt=0.5, M=M, n=10, model=RuleClassifier,rule=["A","B"])

# xt = 0.6
MeanA3, DevA3, MeanB3, DevB3 = calRealError(xt=0.6, M=M, n=10, model=RuleClassifier,rule=["A","B"])

# Plot the changes of deviation with increasing M = [10,100,1000,1000]
plt.plot(M,DevA1, label = "xt = 0.2")
plt.plot(M,DevA2, label = "xt = 0.5")
plt.plot(M,DevA3, label = "xt = 0.6")
plt.legend()
plt.show()


## 1.3 Baseline Classiers (2 points)
# Calculate the 'real' error rate by repeating 10 test datasets of the same size M = [10,100,1000,1000]
M = [10,100,1000,10000]
MeanC, DevC, MeanD, DevD = calRealError(xt=None,M=M, n=10, model=RuleClassifier,rule=["C","D"])

# Plot the changes of deviation with increasing M = [10,100,1000,1000]
plt.plot(M,MeanC, label = "Rule C Mean")
plt.plot(M,DevC, label = "Rule C Std")
plt.plot(M,MeanD, label = "Rule D Mean")
plt.plot(M,DevD, label = "Rule D Std")
plt.legend()
plt.show()


## 1.4 Nearest Neighbor Classication (6 points)

def create_NN_data(N):

    # Generate Y and ensure at least one instance of either class
    Y = np.random.choice([0,1], size = N, p = [0.5,0.5])   # shape:(N,)
    while len(set(Y)) == 1:
        Y = np.random.choice([0,1], size = N, p = [0.5,0.5])   # shape:(N,)

    # Initialize X 
    X = np.array([])   

    for y in Y:

        # Generate uniformly distributed sample for inverse CDF method
        u = np.random.uniform(0, 1)

        # Generate X using inverse CDF method
        if y == 0 : 
            X = np.append( X, 1 - np.sqrt(1-u) )
        else :
            X = np.append( X, np.sqrt(u) )
    
    return X,Y


def NNclassifier(x_train, y_train, x_test):

    y_test = []
    for xi in x_test:
        y = y_train[ (np.abs(x_train - xi)).argmin() ]
        y_test = np.append(y_test, y)

    return y_test



def NNclassifier_xt(x_train, y_train, x_test):

    y_test = []
    x0 = np.mean(x_train[y_train == 0])
    x1 = np.mean(x_train[y_train == 1])
    xt = abs((x0 + x1) / 2 )
    
    if x0 < x1:
        for xi in x_test:
            y = xi - xt
            y_test = np.append(y_test, y)


    return y_test




def calNNError(N_train, N_test, n, model):

    MeanA , DevA = [], []

    for m in N_train:
        
        X_test,Y_test = create_NN_data(N_test)

        errorA = []
        for _ in range(n):
            X, Y = create_NN_data(m)
            y_test = model(x_train=X, y_train=Y, x_test=X_test)
            errorA.append( (N_test - np.sum(y_test == Y_test)) / N_test )

        meanErrorA, devErrorA = np.mean((errorA)),np.std(errorA)

        MeanA.append(meanErrorA)
        DevA.append(devErrorA)

        print("Nearest Neighbor Classifier--> Trainset N =",m," : mean =",round(meanErrorA,2)," ; ","sd =",round(devErrorA,3))

    return MeanA, DevA


# Set train dataset size N_train = [2,100]
# Set test dataset size suffciently large N_test = 1000
N_train = [2,100]
N_test = 1000
# Repeat each experiment (N_train = [2,100]) with 100 different train set
n = 100
Mean, Dev = calNNError(N_train=N_train, N_test=N_test, n=100, model=NNclassifier)







