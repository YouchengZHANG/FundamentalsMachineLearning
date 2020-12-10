
## Exercise 3

# 1 Comment on your solution to exercise 2b

# 2 Comment on others' solution to Exercise 2b


# 3 LDA-Derivation from the Least Squares Error (24 points)

# 3.1 
# Compute b^
'''
2 * E[i=1,N](W.T * xi + b + yi) * 1 = 0
W.T * E(xi) + N * b + E(yi) = 0
W.T * 1/2 * N * (mu0 + mu1) + N * b + 1/2 * (-1 * N + 1 * N) = 0
b^ = - 1/2 * W.T * (mu0 + mu1)
'''

# 3.2 
# reshuffle W to intermediate equation
'''
E[i=1,N](W.T * xi + b + yi) * xi = 0
E(xi*xi.T*W) - 1/2*E(xi*(mu0+mu1).T*W) - 1/2*N*(mu1-mu0) = 0
1/N * W*E(xi*xi.T) - 1/4*(mu0+mu1)*(mu0+mu1).T = 1/2 * (mu1-mu0)    --- (1)  

∵   Sw + 1/4*Sb = 1/N * E((xi-muyi)(xi-muyi).T) + 1/4 * E((mu1-mu0)(mu1-mu0).T)   --- (2)
∴   Here we need to prove LHS of Eq(1) ~ RHS of Eq(2)

1/N * E((xi-muyi)(xi-muyi).T) + 1/4 * E((mu1-mu0)(mu1-mu0).T)
1/N*E(xi*xi.T) - 1/N*E(muyi*xi.T) - 1/N(xi*muyi.T) + 1/N*E(muyi*muyi.T) + 1/4*(mu0*mu0.T+mu1*mu1.T-mu0*mu1.T-mu1*mu0.T)   --- (3)

∵   1/N*(xi*muyi.T) = 1/N*E(muyi*xi.T) = 1/2*(mu0*mu0.T+mu1*mu1.T)    --- (4)
∴   Expand Eq(3) with Eq(4)

1/N * W*E(xi*xi.T) - 1/4*(mu0+mu1)*(mu0+mu1).T = 1/N * W*E(xi*xi.T) - 1/4*(mu0+mu1)*(mu0+mu1).T  = Sw + 1/4*Sb
'''

# 3.3
# Compute W^
'''
(Sw + 1/4*Sb)W =  1/2 * (mu1 - mu0)   --- (5)
τ is an arbitrary positive constant， expressing the sign of τ term is the same direction of the origin term

∵   Sb = (mu1 - mu0)(mu1 - mu0).T
    Sb*W = (mu1 - mu0)(mu1 - mu0).T * W 
    (mu1 - mu0).T * W = (mu1 - mu0).T * Kw * (mu1 - mu0) ~ C where C is a constant with shape (1,1)
    Sb*W = C * (mu1 - mu0) 
∴   Simpily Eq(5)
    Sw * W = (1/2 - 1/4*C)(mu1 - mu0)    --- (6)

If the number of instances is NOT much less then the number of features
Sw scatter matrix is invertible, so that Sw^-1 * Sw * W = Sw^-1 * A
Thus, simpilfy Eq(6)
∴   Sw^-1 * Sw * W = Sw^-1 * (1/2 - 1/4*C)(mu1 - mu0)
    W = Sw^-1 * (1/2 - 1/4*C)(mu1 - mu0)
    W = τ * Sw^-1 * (mu1 - mu0)      
    
'''

# 4 Data Generation with QDA (8 points)
import matplotlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn import model_selection
import matplotlib.pyplot as plt

digits = load_digits()
print(digits.keys())

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

X_sub = data[ (target== 1)|(target== 7) , : ]
Y_sub = target[ (target== 1)|(target== 7) ]

np.shape(X_sub)
np.shape(Y_sub)

X_train , X_test , Y_train , Y_test = model_selection.train_test_split(X_sub, Y_sub,test_size = 0.4, random_state = 0)

def fit_qda(training_features, training_labels):

    training_labels = (training_labels == list(set(training_labels))[0]).__invert__().astype(int)
    TL = set(training_labels)   # [1,7]
    p = [len(training_features[training_labels==tl,:])/len(training_features) for tl in TL]
    mu = np.array([np.mean(training_features[training_labels==tl, : ], axis=0) for tl in TL])  # ==1,==7  ,  (2,2)
    covmat = np.array([np.cov(training_features[training_labels == tl,:].T) for tl in TL])

    return mu, covmat, p

mu, covmat, p = fit_qda(training_features=X_train, training_labels=Y_train)

# Generate 8 instances for each class using multivariate_normal
n = 8

# Generate digit 1 and plot
k = 0
inst = np.random.multivariate_normal(mu[k,], covmat[k,:], (8,))

plt.figure()
fig, ax = plt.subplots(nrows=2,ncols=4)
ax = ax.reshape(8,)
for i in range(n):
    ax[i,].imshow(inst[i,:].reshape(8,8), interpolation="nearest") 
plt.show()

# Generate digit 7 and plot
k = 1
inst = np.random.multivariate_normal(mu[k,], covmat[k,:], (8,))

plt.figure()
fig, ax = plt.subplots(nrows=2,ncols=4)
ax = ax.reshape(8,)
for i in range(n):
    ax[i,].imshow(inst[i,:].reshape(8,8), interpolation="nearest") 
plt.show()



