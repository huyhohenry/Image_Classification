import numpy as np
from numba import jit

class WeakClassifier():
    """ weak classifier - threshold on the features
    Args:
        X (numpy.array): data array of flattened images
                        (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (num observations, )
    """
    def __init__(self, X, y, weights, thresh=0, feat=0, sign=1):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)

        self.idx_0  = self.ytrain == -1
        self.idx_1  = self.ytrain == 1
        self.threshold = thresh
        self.feature = feat
        self.sign = sign
        self.weights = weights

    def train(self):
        # save the threshold that leads to best prediction
        tmp_signs = []
        tmp_thresholds = []

        for f in range(self.Xtrain.shape[1]):
            m0 = self.Xtrain[self.idx_0, f].mean()
            m1 = self.Xtrain[self.idx_1, f].mean()
            tmp_signs.append(1 if m0 < m1 else -1)
            tmp_thresholds.append((m0+m1)/2.0)

        tmp_errors=[]
        for f in range(self.Xtrain.shape[1]):
            tmp_result = self.weights*(tmp_signs[f]*((self.Xtrain[:,f]>tmp_thresholds[f])*2-1) != self.ytrain)
            tmp_errors.append(sum(tmp_result))

        feat = tmp_errors.index(min(tmp_errors))

        self.feature = feat
        self.threshold = tmp_thresholds[feat]
        self.sign = tmp_signs[feat]
        # -- print(self.feature, self.threshold)

    def predict(self, x):
        return self.sign * ((x[self.feature] > self.threshold) * 2 - 1)

@jit(nopython=True)
def train_optimized(Xtrain, ytrain, weights):
    signs = [1.] * Xtrain.shape[1]
    thresholds = [0.] * Xtrain.shape[1]
    errors = [100.] * Xtrain.shape[1]
    min_e = 10000000000000
    for f in range(Xtrain.shape[1]):
        tmp_thresholds = Xtrain[:,f].copy()
        tmp_thresholds = np.unique(tmp_thresholds)
        tmp_thresholds.sort()

        tmp_thresholds = [(tmp_thresholds[i]+tmp_thresholds[i+1])/2 for i in
                            range(len(tmp_thresholds)-1)]

        for theta in tmp_thresholds:
            for s in [1,-1]:
                tmp_r = weights * ( s*((Xtrain[:,f]<theta)*2-1) != ytrain)
                tmp_e = np.sum(tmp_r, np.float64)
                if tmp_e < min_e:
                    feature = f
                    threshold = theta
                    polarity = s
                    error = tmp_e
                    min_e = tmp_e

    if error == 0: error = 1e-6
    elif error == 1: error = 1 - 1e-6

    return feature, threshold, polarity, error

class VJ_Classifier:
    """Weak classifier for Viola Jones procedure

    Args:
        X (numpy.array): Feature scores for each image. Rows: number of images
                         Columns: number of features.
        y (numpy.array): Labels array of shape (num images, )
        weights (numpy.array): observations weights array of shape (num observations, )

    Attributes:
        Xtrain (numpy.array): Feature scores, one for each image.
        ytrain (numpy.array): Labels, one per image.
        weights (float): Observations weights
        threshold (float): Integral image score minimum value.
        feat (int): index of the feature that leads to minimum classification error.
        polarity (float): Feature's sign value. Defaults to 1.
        error (float): minimized error (epsilon)
    """
    def __init__(self, X, y, weights, thresh=0, feat=0, polarity=1):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.weights = weights
        self.threshold = thresh
        self.feature = feat
        self.polarity = polarity
        self.error = 0

    def train(self):
        """Trains a weak classifier that uses Haar-like feature scores.

        This process finds the feature that minimizes the error as shown in
        the Viola-Jones paper.

        Once found, the following attributes are updated:
        - feature: The column id in X.
        - threshold: Threshold (theta) used.
        - polarity: Sign used (another way to find the parity shown in the
                    paper).
        - error: lowest error (epsilon).
        """
        self.feature, self.threshold, self.polarity, self.error = train_optimized(self.Xtrain, self.ytrain, self.weights)

    def predict(self, x):
        """Returns a predicted label.

        Inequality shown in the Viola Jones paper for h_j(x).

        Args:
            x (numpy.array): Scores obtained from Haar Features, one for each
                             feature.

        Returns:
            float: predicted label
        """
        return self.polarity * ((x[self.feature] < self.threshold) * 2 - 1)
