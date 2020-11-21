
## Exercise 1b

# 1 Comment on your solution to Exercise 1a
# markdown cells starting with <span style="color:green;font-weight:bold">Comment</span>


# 2 Comment on others' solution to Exercise 1a
# markdown cells starting with <span style="color:green;font-weight:bold">Comment</span>



## 3 Nearest Neighbor Classi􏰃cation on Real Data
# 3.1 Exploring the Data (3 points)

import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.keys())

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]
print(data.dtype)
#print(digits['DESCR'])


# print the size of images
# the number of sample = 1797
# the size of each image = (8,8)
np.shape(data)
np.shape(images)


# Visualize one image of a 3 using the imshow
import matplotlib.pyplot as plt
img = images[3]
assert 2 == len(img.shape)
plt.figure()
plt.gray()
plt.imshow(img, interpolation="nearest") 
plt.show()

# Visualize by changing interpolation="bicubic" 
plt.imshow(img, interpolation="bicubic")
plt.show()


# Split data into train/test set
from sklearn import model_selection
X_all = data 
Y_all = target
X_train , X_test , Y_train , Y_test = model_selection.train_test_split(X_all, Y_all,test_size = 0.4, random_state = 0)


# 3.2 Distance function computation using loops (3 points)
# Use the X_train and X_test as input
def dist_loop(training, test):

    dist = [ np.linalg.norm(xj-xi)  for xj in training  for xi in test ]
    distm = np.array(dist, dtype=np.float64).reshape( (len(training),len(test)) )

    return distm

distm1 = dist_loop(training=X_train, test=X_test)
len(X_train), len(X_test)
np.shape(distm1)

# Measure the run time using dist_loop
%timeit dist_loop(training=X_train, test=X_test)


# 3.3 Distance function computation using vectorization (8 points)
def dist_vec(training, test):

    distm =  np.linalg.norm(training[ : , np.newaxis] - test, axis = 2)

    return distm

distm2 = dist_vec(training=X_train, test=X_test)
len(X_train), len(X_test)
np.shape(distm2)

# Measure the run time using dist_vec
%timeit dist_vec(training=X_train, test=X_test)


# 3.4 Implement the k-nearest neighbor classi􏰃er (6 points)
def KNNclassifier(x_train, y_train, x_test, K):

    y_test = []

    distm = dist_vec(training = x_train, test = x_test)
    disti = np.argpartition(distm, kth = K, axis = 0)[ :K , : ]
    distv = y_train[disti]
    y_test = np.array([ np.bincount(distv[ : , v ]).argmax() for v in range(len(distv[0])) ])

    return y_test


def calKNNError(x_train, y_train, x_test, Y_test, K, model, stdout):

    errorA = []
    for k in K:

        if model == KNNclassifier:
            y_test = model(x_train=x_train, y_train=y_train, x_test=x_test, K=k)
        elif model == KNeighborsClassifier:
            y_test = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train).predict(x_test)

        N_test = len(Y_test)
        err = (N_test - np.sum(y_test == Y_test)) / N_test
        errorA.append( err )

        if stdout == True:
            print( "K Nearest Neighbor Classifier--> K =",k," : error rates =",round(err,5) )

    return errorA


# Subset the data to digit 3 and 9
# Split the subset data to train/test set
X_sub = data[ (target== 3)|(target== 9) , : ]
Y_sub = target[ (target== 3)|(target== 9) ]
X_train , X_test , Y_train , Y_test = model_selection.train_test_split(X_sub, Y_sub,test_size = 0.4, random_state = 0)

# Set different K = [1, 3, 5, 9, 17, 33]
K = [1, 3, 5, 9, 17, 33]

# Compute the error rate with different K
errorKNN = calKNNError(x_train=X_train, y_train=Y_train, x_test=X_test, Y_test=Y_test, K=K, model=KNNclassifier, stdout=True)

# Plot the dependency of the classification performance on K
plt.plot(K,errorKNN, label = "K")
plt.show()



# 4 Cross-validation (8 points)
def split_folds(data, target, L):
    idx = np.random.permutation(len(data))
    idx_sp = np.array_split(idx,L)

    X_folds = [ data[i,:] for i in idx_sp ] 
    Y_folds = [ target[i,] for i in idx_sp ]  

    return X_folds, Y_folds

def evaluateKNN(data, target, L, K, model, stdout):

    MeanL = []
    DevL = []
    for ls in L:
        X_folds, Y_folds = split_folds(data=data, target=target, L=ls)
        errorK = []
        for l in range(ls):
            lx = [ x for x in range(ls) if x != l ]
            X_test = X_folds[l]
            Y_test = Y_folds[l]

            X_train = np.concatenate( [ X_folds[i] for i in lx ] , axis=0 )
            Y_train = np.concatenate( [ Y_folds[i] for i in lx ] , axis=0 )

            errorKNN = calKNNError(x_train=X_train, y_train=Y_train, x_test=X_test, Y_test=Y_test, K=K, model=model, stdout=stdout)[0]
            errorK.append(errorKNN)
        if stdout == True:
            print("KNN classifier -->","Fold =",ls,": Error rate Mean =",round(np.mean(errorK),3),"; Std =",round(np.std(errorK),5))
        
        MeanL.append(np.mean(errorK))
        DevL.append(np.std(errorK))

    #errorKL = [ [ errorL[l][k]  for l in range(len(L))] for k in range(len(K)) ]
    return MeanL, DevL

# Randomly split the given data and labels into L folds, here take L = 3
X_folds, Y_folds = split_folds(data=data, target=target, L=3)
X_folds

# Evaluate the KNN performance on full datasets over the L repetitions
# Set L = [2,5,10]
L = [2,5,10]

# Set K = [1]
MeanL1, DevL1 = evaluateKNN(data=data, target=target, L=[2,5,10], K=[1], model=KNNclassifier, stdout=True)
%timeit evaluateKNN(data=data, target=target, L=[2,5,10], K=[1], model=KNNclassifier, stdout=False)

# Set a bigger K = [33]
MeanL33, DevL33 = evaluateKNN(data=data, target=target, L=[2,5,10], K=[33], model=KNNclassifier, stdout=True)
%timeit evaluateKNN(data=data, target=target, L=[2,5,10], K=[33], model=KNNclassifier, stdout=False)

# Plot the dependency of the classification performance on L using KNNclassifier
plt.plot(L,MeanL1, label = "error rate Mean (K=1)")
plt.plot(L,DevL1, label = "error rate Std (K=1)")
plt.plot(L,MeanL33, label = "error rate Mean (K=33)")
plt.plot(L,DevL33, label = "error rate Std (K=33)")
plt.legend()
plt.show()


# Use sklearn library method: sklearn.neighbors.KNeighborsClassifier()
from sklearn.neighbors import KNeighborsClassifier

# Set K = [1]
MeanL1_skt, DevL1_skt = evaluateKNN(data=data, target=target, L=[2,5,10], K=[1], model=KNeighborsClassifier, stdout=True)
%timeit evaluateKNN(data=data, target=target, L=[2,5,10], K=[1], model=KNeighborsClassifier, stdout=False)

# Set a bigger K = [33]
MeanL33_skt, DevL33_skt = evaluateKNN(data=data, target=target, L=[2,5,10], K=[33], model=KNeighborsClassifier, stdout=True)
%timeit evaluateKNN(data=data, target=target, L=[2,5,10], K=[33], model=KNeighborsClassifier, stdout=False)

# Plot the dependency of the classification performance on L using sklearn library
plt.plot(L,MeanL1_skt, label = "skt error rate Mean (K=1)")
plt.plot(L,DevL1_skt, label = "skt error rate Std (K=1)")
plt.plot(L,MeanL33_skt, label = "skt error rate Mean (K=33)")
plt.plot(L,DevL33_skt, label = "skt error rate Std (K=33)")
plt.legend()
plt.show()


