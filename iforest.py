
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics import confusion_matrix
import multiprocessing as mp
from scipy.stats import kurtosis

class ExNode:
    def __init__(self, size):
        self.size = size
        self.left = None
        self.right = None


class InNode:
    def __init__(self, left, right, q, p):
        self.left = left
        self.right = right
        self.q = q
        self.p = p


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees):
        
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []
        self.height_limit = math.ceil(math.log(self.sample_size,2))
    
    def fit(self, X: np.ndarray, improved=False):
        """
            Given a 2D matrix of observations, create an ensemble of IsolationTree
            objects and store them in a list: self.trees.  Convert DataFrames to
            ndarray objects.
            """
        
        if isinstance(X, pd.DataFrame):
            X = X.values
    
        for i in range(self.n_trees):
            X_1 = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
            iTree = IsolationTree(X_1, self.height_limit,improved)
            self.trees.append(iTree)
        
        return self
    
    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
            Given a 2D matrix of observations, X, compute the average path length
            for each observation in X.  Compute the path length for x_i using every
            tree in self.trees then compute the average for each x_i.  Return an
            ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
    
        def single_tree_length(x, Tree, current_height = 0):
            if isinstance(Tree, ExNode):
                return current_height
            a = Tree.q
            if x[a] < Tree.p:
                return single_tree_length(x, Tree.left, current_height + 1)
            else:
                return single_tree_length(x, Tree.right, current_height + 1)


        total_length = []
        for x_i in X:
            single_length = []
            for tree in self.trees:
                single_length.append(single_tree_length(x_i, tree.root, 0))
            total_length.append(np.mean(single_length))
        return np.array(total_length).reshape(-1, 1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
            Given a 2D matrix of observations, X, compute the anomaly score
            for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
    
                        
        def c(x):
            if x > 2:
                return 2 * (np.log(x-1) + 0.5772156649) - 2 * (x - 1) / x
            if x == 2:
                return 1
            else:
                return 0
                                                    
        n = X.shape[1]
        return 2 ** (-self.path_length(X) /c(n))
        
    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        
        return (scores>=threshold).astype(int)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."""
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)



class IsolationTree:
    def __init__(self, X, height_limit, improved =False):
        
        self.X = X
        self.h = height_limit
        self.root = self.fit(X, self.h, improved)
        self.n_nodes = self.nodes(self.root)
    
    def fit(self, X: np.ndarray, height_limit, improved=False):
        """
            Given a 2D matrix of observations, create an isolation tree. Set field
            self.root to the root of that tree and return it.
            
            If you are working on an improved algorithm, check parameter "improved"
            and switch to your new functionality else fall back on your original code.
            """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if height_limit == 0 or len(X) <= 1: return ExNode(len(X))
        if improved == True :
            Q  = np.random.choice(np.arange(X.shape[1]),4)
            d= []
            pairs = []
            for q in Q :
                max_num = max(X[:,q])
                min_num = min(X[:,q])
                p = np.random.uniform(min_num, max_num)
                Xl = X[X[:, q] < p]
                Xr = X[X[:, q] >= p]
                d.append(min(len(Xl),len(Xr)))
                pairs.append((q,p))
            q,p =pairs[np.argmin(d)]
            max_num = max(X[:,q])
            min_num = min(X[:,q])
            if max_num == min_num:
                return ExNode(len(X))
            Xl = X[X[:, q] < p]
            Xr = X[X[:, q] >=  p]
            return InNode(self.fit(Xl, height_limit - 1, improved), self.fit(Xr, height_limit - 1,improved), q, p)
        if improved == False :
            Q = X.shape[1]
            q = np.random.choice(np.arange(Q))
            max_num = max(X[:,q])
            min_num = min(X[:,q])
            if max_num == min_num:
                return ExNode(len(X))
            p = np.random.uniform(min_num, max_num )
            Xl = X[X[:, q] < p]
            Xr = X[X[:, q] >= p]
            return InNode(self.fit(Xl, height_limit - 1, improved), self.fit(Xr, height_limit - 1,improved), q, p)

    def nodes(self, Tree):
        if not Tree:return 0
        else : return 1 + self.nodes(Tree.left) + self.nodes(Tree.right)


def find_TPR_threshold(y, scores, desired_TPR):
    """
        Start at score threshold 1.0 and work down until we hit desired TPR.
        Step by 0.01 score increments. For each threshold, compute the TPR
        and FPR to see if we've reached to the desired TPR. If so, return the
        score threshold and FPR.
    """
    thresholds = [i * 0.01 for i in range(100)]
    for threshold in thresholds[::-1]:
        y_pred = (scores>=threshold).astype(int)
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return threshold, FPR
