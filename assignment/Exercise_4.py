
## Exercise 4

# 1 Comment on your and other’s solution to Exercise 3
# 2 Density Tree and Decision Tree (20 points)



# Leave-one-out error : https://www.statology.org/leave-one-out-cross-validation-in-python/
# https://www.statology.org/leave-one-out-cross-validation/



# Preliminaries

# import modules
import numpy as np
from numpy.compat.py3k import npy_load_module
# your code here

# base classes

class Node:
    pass

class Tree:
    def __init__(self):
        self.root = Node()
    
    def find_leaf(self, x):
        node = self.root
        while hasattr(node, "feature"):
            j = node.feature
            if x[j] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node



# Density Tree

class DensityTree(Tree):
    def __init__(self):
        super(DensityTree, self).__init__()
        
    def train(self, data, prior, n_min=20):
        '''
        data: the feature matrix for the digit under consideration
        prior: the prior probability of this digit
        n_min: termination criterion (don't split if a node contains fewer instances)
        '''
        self.prior = prior
        N, D = data.shape
        D_try = int(np.sqrt(D)) # number of features to consider for each split decision

        # find and remember the tree's bounding box, 
        # i.e. the lower and upper limits of the training feature set
        m, M = np.min(data, axis=0), np.max(data, axis=0)
        self.box = m.copy(), M.copy()
        
        # identify invalid features and adjust the bounding box
        # (If m[j] == M[j] for some j, the bounding box has zero volume, 
        #  causing divide-by-zero errors later on. We must exclude these
        #  features from splitting and adjust the bounding box limits 
        #  such that invalid features have no effect on the volume.)
        valid_features   = np.where(m != M)[0]
        invalid_features = np.where(m == M)[0]
        M[invalid_features] = m[invalid_features] + 1

        # initialize the root node
        self.root.data = data
        self.root.box = m.copy(), M.copy()

        # build the tree
        stack = [self.root]
        while len(stack):
            node = stack.pop()   # pop the last term in the list
            n = node.data.shape[0] # number of instances in present node
            if n >= n_min:
                # Call 'make_density_split_node()' with 'D_try' randomly selected 
                # indices from 'valid_features'. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                feature_indices = np.random.choice(valid_features, size=D_try, replace=False)
                left, right = make_density_split_node(node=node, N=n, feature_indices=feature_indices)
                stack += [left,right]
                
            else:
                # Call 'make_density_leaf_node()' to turn 'node' into a leaf node.
                make_density_leaf_node(node=node, N=n)

    def predict(self, x):
        leaf = self.find_leaf(x)
        # return p(x | y) * p(y) if x is within the tree's bounding box 
        # and return 0 otherwise
        return ... # your code here





def make_density_split_node(node, N, feature_indices):
    '''
    node: the node to be split
    N:    the total number of training instances for the current class
    feature_indices: a numpy array of length 'D_try', containing the feature 
                     indices to be considered in the present split
    '''
    n, D = node.data.shape                              # self.root.data
    m, M = node.box   # m: min values ; M: max values   # self.root.box

    # find best feature j (among 'feature_indices') and best threshold t for the split
    e_min = float("inf")
    j_min, t_min = None, None
    
    for j in feature_indices:
        # Hint: For each feature considered, first remove duplicate feature values using 
        # 'np.unique()'. Describe here why this is necessary.

        # Duplicated terms might confuse the splitting processes which cannot determine which side should be assigned
        # since the mean of duplicated value will be itself.
        # Also, np.unique() can automatically sort the value.

        data_unique = np.unique((node.data[:, j]))     # N x 1
        # Compute candidate thresholds
        tj = np.mean([data_unique[:-1],data_unique[1:]], axis=0)     
        
        # Illustration: for loop - hint: vectorized version is possible
        for t in tj:
            
            mL, ML = m.copy(), M.copy()
            ML[:,j] = t
            mR, MR = m.copy(), M.copy()
            mR[:,j] = t

            VL = np.prod(ML - mL)
            VR = np.prod(MR - mR)
            
            NL = node.data[node.data[:,j] < t, : ].shape[0]
            NR = node.data[node.data[:,j] > t, : ].shape[0]

            errL = (NL/(N*VL)) * (NL/N - 2*((NL-1)/(N-1)))
            errR = (NR/(N*VR)) * (NR/N - 2*((NR-1)/(N-1)))

            # Compute the error
            loo_error = errL + errR
            
            # choose the best threshold that
            if loo_error < e_min:
                e_min = loo_error
                j_min = j
                t_min = t


    # create children
    left = Node()
    right = Node()

    #####  np.shares_memory()
    # initialize 'left' and 'right' with the data subsets and bounding boxes
    # according to the optimal split found above
    m_new, M_new = m.copy(), M.copy()
    M_new[:,j_min], m_new[:,j_min]= t_min, t_min  # New    #m_new[:,j_min] = t_min
    left.data =  node.data[node.data[:,j_min] < t_min , :] # store data in left node -- for subsequent splits
    left.box = node.box[0].copy(), M_new.copy()  # store bounding box in left node
    right.data = node.data[node.data[:,j_min] > t_min , :]
    right.box = m_new.copy(), node.box[1].copy()

    # turn the current 'node' into a split node
    # (store children and split condition)
    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min

    # return the children (to be placed on the stack)
    return left, right


def make_density_leaf_node(node, N):
    '''
    node: the node to become a leaf
    N:    the total number of training instances for the current class
    '''
    # compute and store leaf response
    n = node.data.shape[0]
    v = np.prob(node.box[1] - node.box[0])
    node.response = n/(N*v)



# Decision Tree

class DecisionTree(Tree):
    def __init__(self):
        super(DecisionTree, self).__init__()
        
    def train(self, data, labels, n_min=20):
        '''
        data: the feature matrix for all digits
        labels: the corresponding ground-truth responses
        n_min: termination criterion (don't split if a node contains fewer instances)
        '''
        N, D = data.shape
        D_try = int(np.sqrt(D)) # how many features to consider for each split decision

        # initialize the root node
        self.root.data = data
        self.root.labels = labels
        
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0] # number of instances in present node
            if n >= n_min and not node_is_pure(node):
                # Call 'make_decision_split_node()' with 'D_try' randomly selected 
                # feature indices. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                
            else:
                # Call 'make_decision_leaf_node()' to turn 'node' into a leaf node.
                ... # your code here
                
    def predict(self, x):
        leaf = self.find_leaf(x)
        # compute p(y | x)
        return ... # your code here


def make_decision_split_node(node, feature_indices):
    '''
    node: the node to be split
    feature_indices: a numpy array of length 'D_try', containing the feature 
                     indices to be considered in the present split
    '''
    n, D = node.data.shape
    e_min = float("inf")
    j_min, t_min = None, None

    # find best feature j (among 'feature_indices') and best threshold t for the split
    for j in feature_indices:
        data_unique = np.unique((node.data[:, j]))     # N x 1
        # Compute candidate thresholds
        tj = np.mean([data_unique[:-1],data_unique[1:]], axis=0)     
        
        # Illustration: for loop - hint: vectorized version is possible
        for t in tj:
            
            NL = node.labels[node.data[:,j] < t].shape[0]
            NR = node.labels[node.data[:,j] > t].shape[0]

            errGL = NL*(1-sum( [(NL[NL==list(set(digits))[k]] / NL) ** 2 for k in list(set(node.labels))] ))
            errGR = NR*(1-sum( [(NR[NR==list(set(digits))[k]] / NR) ** 2 for k in list(set(node.labels))] ))

            # Compute the error
            gini_error = errGL + errGR
            
            # choose the best threshold that
            if gini_error < e_min:
                e_min = gini_error
                j_min = j
                t_min = t

    # create children
    left = Node()
    right = Node()

    # initialize 'left' and 'right' with the data subsets and labels
    # according to the optimal split found above
    left.data = node.data[node.data[:,j_min] < t_min]
    left.labels = node.labels[node.data[:,j_min] < t_min]
    right.data = node.data[node.data[:,j_min] > t_min]
    right.labels = node.labels[node.data[:,j_min] < t_min]

    # turn the current 'node' into a split node
    # (store children and split condition)
    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min

    # return the children (to be placed on the stack)
    return left, right    



def make_decision_leaf_node(node):
    '''
    node: the node to become a leaf
    '''
    # compute and store leaf response
    node.N = ...
    node.response = ... # your code here


def node_is_pure(node):
    '''
    check if 'node' ontains only instances of the same digit
    '''
    return len(set(node.labels)) == 1



# Evaluation of Density and Decision Tree



















digits


len(set(digits))
set(digits).tolist()[0]

list(set(digits))

digits = [10, 20, 30, 40]
10**2
np.where(digits[2],99,digits[:])
digits = np.array(digits)


digits[-1: -4]
digits[-1: -4 : -1]

digits[1:]
digits[:-1]

import numpy as np
np.vstack([digits[1:],digits[:-1]])


np.unique([1,2,5,6,6,6,3,3,2,4])


(digits[0:-1] + digits[1:]) / 2


np.mean((digits[0:-1],digits[1:]),axis=0)


np.mean()

(digits[0:-1],digits[1:])

M = [15,5]
m = 2

M = [15]
M.append(55)

Ml = M
ml = m

float("inf")
e_min = float("inf")

-10 < e_min


class Node:
    pass

test = Node()
test.wtf = [1,2,3,4,5]


np.array([[1,4,5,6,7,20],[2,4,15,25,9,22]]).shape


np.max(np.array([[1,4,5,6,7,20],[2,4,15,25,9,22]]), axis=0)

a = 100
b = 2000

digits.append([a,b])
digits.append(left)
digits.append(right)

digits.extend([a,b])

digits += [a,b]

digits += [left,right]

