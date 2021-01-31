

## Exercise 6

# 1 Comment on your and other's solution to Exercise 5
# 2 Bias and variance of ridge regression (8 points)


# 3 Denoising of a CT image (11 points)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt

def construct_X(M, alphas, Np = None, tau=0):

    # M: input size, e.g. tissue size
    # Np: sensor maximum/desirable size
    # N: sensor desirable size * the numebr of rotation angle
    # X: output with shape [ N , D ]

    M = M + 1 if M % 2 == 0 else M
    Np = int(np.sqrt(2) * M) + 1 if Np == None else Np
    D = M * M 
    N = Np * len(alphas)
    
    # create an array C ∈ R[2×D] holding the coordinates of the tomogram's pixel centers
    c = (M - 1) / 2
    C1, C0 = np.mgrid[-c:c+1, -c:c+1]
    C1 = np.flip(C1)
    C = np.stack((C0,C1)).reshape(2,D)

    # the projection p of each pixel onto the sensor 
    # p = c0 * cos + c1 * sin
    # s0 is the distance between the first sensor element and the sensor's coordinate origin
    r = np.radians(alphas)
    n = np.stack((np.cos(r), np.sin(r))).reshape(2,len(r))
    s0 = (Np - 1) / 2 
    p = np.dot(n.T, C) + s0
    
    p_frac, p_int = np.modf(p)

    row = (p_int + Np * np.arange(len(alphas)).reshape(len(alphas),-1)).flatten()
    col = np.tile(np.arange(D),(len(alphas,)))
    weights = 1 - p_frac.flatten()

    row = np.append(row,row+1)
    col = np.append(col,col)
    weights = np.append(weights,1-weights)

    X = sparse.coo_matrix((weights, (row, col)), shape=(N, D), dtype = np.float32)
    
    if tau > 0: 
        Itau = sparse.coo_matrix(np.eye(D) * np.sqrt(tau))
        Xtau = sparse.vstack((X,Itau))
        return Xtau

    return X



# Reconstruct the tomogram for 64 angles with τ = 0,1,10,100,1000,10000
Y_195 = np.load("hs_tomography/y_195.npy")
beta_195 = np.load("hs_tomography/beta_195.npy")
alphas_195 = np.load("hs_tomography/alphas_195.npy")

ind = np.append(np.append(np.arange(3,89,3),np.arange(89,177)[::-3]),[0,1,2,177,178])
alphasS_195 = sorted(alphas_195[ind])

M = 195
Np = 275
YS_195 = np.zeros((len(ind) * Np + M*M))
for i in range(len(ind)):
    YS_195[Np * i : Np * (i+1)] = Y_195[Np * ind[i] : Np * (ind[i] + 1)] 


print("The number of reduced angles: ",len(alphasS_195))
print("The length of Y: ",len(YS_195))

tauL = [0,1,10]
fig, ax = plt.subplots(2,3)
ax = ax.reshape(1,-1)

for t in range(len(tauL)): 
    XS_195 = construct_X(195, alphasS_195, Np, tau=tauL[t])
    print("Reconstruction Finished")
    betaS_195 = lsqr(XS_195, YS_195, atol = 1e-04, btol = 1e-04)

    imgS_195 = betaS_195[0].reshape(195,195)
    ax[0][t].imshow(imgS_195, interpolation = "none")

plt.show()



# Denoising with Gaussian filtering
from scipy.ndimage import gaussian_filter

Np = 275
YS_195 = np.zeros((len(ind) * Np))
for i in range(len(ind)):
    YS_195[Np * i : Np * (i+1)] = Y_195[Np * ind[i] : Np * (ind[i] + 1)] 

XS_195 = construct_X(195, alphasS_195, Np, tau=0)
betaS_195 = lsqr(XS_195, YS_195, atol = 1e-04, btol = 1e-04)
imgS_195 = betaS_195[0].reshape(195,195)

sigmaL = [0,1,2,3,5,7]
fig, ax = plt.subplots(2,3)
ax = ax.reshape(1,-1)

for s in range(len(sigmaL)): 
    imgG_195 = gaussian_filter(imgS_195, sigma=sigmaL[s])
    print("Gaussian filtering Finished")
    ax[0][s].imshow(imgG_195, interpolation = "none")

plt.show()


# 4 Automatic feature selection for regression
# 4.1 Implement Orthogonal Matching Pursuit (5 points)
# http://users.ece.utexas.edu/~sanghavi/courses/scribed_notes/Lecture_21_Scribe_Notes.pdf
# https://angms.science/doc/RM/OMP.pdf


def omp_regression(X, y, T):

    # X: X ∈ RN×D
    # y: y ∈ RN 
    # T > 0
    # beta^: D × T matrix, where D the initial number of features, T the number of iterations

    A = []
    B = np.arange(0, X.shape[1])
    r = y
    H = np.zeros((X.shape[1],))  

    while len(A) <= T:
        jt = np.argmax(np.dot(X[:,B].T, r))
        A.append(jt)
        B = B[B != jt]

        beta = lsqr(X[:,A], y, atol = 1e-04, btol = 1e-04)[0]
        h = np.zeros((X.shape[1],))
        h[A,] = beta

        H = np.vstack((H,h))
        r = y - np.dot(X[:,A], beta)

    H = H[1:,:]
    return H.T


X = np.random.randint(0,20,200).reshape(20,10)
y = np.random.randint(0,100,20).reshape(20,)
T = 4
solutions = omp_regression(X, y, T)
print(solutions.shape)
solutions


# 4.2 Classi􏰃cation with sparse LDA (8 points)

from sklearn.datasets import load_digits
from sklearn import model_selection

digits = load_digits()
print(digits.keys())

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

X_sub = data[ (target== 1)|(target== 7) , : ]
Y_sub = target[ (target== 1)|(target== 7) ]
Y_sub[Y_sub==7] = -1


solutions = omp_regression(X=X_sub, y=Y_sub, T=64)























####################################


X = np.random.randint(0,20,800).reshape(20,40)
B = np.arange(0,40)
Bx = B[:5]
X[:,[0,1,10,15]]
y = np.random.randint(0,100,20).reshape(20,)
r = y
jt = np.argmax(np.dot(X[:,B].T, r))

A = []
A.append(jt)
B = B[B != jt]

beta = lsqr(X[:,A], y, atol = 1e-04, btol = 1e-04)[0]
r = y - np.dot(X[:,A],beta)


H = np.zeros((X.shape[1]))      #np.empty([40,],dtype=np.float32)
h = np.zeros((X.shape[1]))
h.shape
H.shape
h[A,] = beta

H = np.vstack((H,h))
H


h2 = np.zeros((X.shape[1],))
h2[[0],] = beta
np.concatenate((H,h2), axis=0)



solutions = omp_regression(X, y, T)



###################
    Np = 109
    Ysub_77 = np.zeros((len(sub_ind)*Np))
    for i in range(len(sub_ind)):
        Ysub_77[Np * i : Np * (i+1)] = Y_77[Np * sub_ind[i] : Np * (sub_ind[i] + 1)]

    Xsub_77 = construct_X(77, alphas_sub_77, Np)
    betasub_77 = lsqr(Xsub_77, Ysub_77, atol = 1e-05, btol = 1e-05)
