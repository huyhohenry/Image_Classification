
import numpy as np
import cv2
import os
import math
from helper_classes import WeakClassifier, VJ_Classifier
from numpy.lib.stride_tricks import sliding_window_view

# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder)]
    X = None
    y = None
    for img in images_files:

        img_path = os.path.join(folder, img)
        img_array = cv2.imread(img_path,-1)
        img_resized = cv2.resize(img_array,size)
        if X is None:
            X = np.ndarray.flatten(img_resized)
        else:
            X = np.vstack((X,np.ndarray.flatten(img_resized)))

        if y is None:
            y = int(str(img)[7:9])
        else:
            y = np.vstack((y,int(str(img)[7:9])))
    y = y.reshape((np.shape(y)[0],))

    return (X,y)

def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    options = np.arange(np.size(y),dtype = int)
    number_of_training_set = math.floor(np.size(y) * p)
    options_randomized = np.random.permutation(options)
    picked_options = options_randomized[:number_of_training_set]
    tf = np.isin(options, picked_options)
    Xtrain = np.compress(tf, X,axis=0)
    ytrain = np.compress(tf, y)
    Xtest = np.compress(~tf, X,axis=0)
    ytest = np.compress(~tf, y)
    return (Xtrain, ytrain, Xtest, ytest)


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    m = np.average(x,0)
    return np.average(x,0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    mean_face = get_mean_face(X)
    A = np.subtract(X,mean_face)
    C = np.matmul(np.transpose(A),A)

    C_vectorized = np.reshape(C,(int(np.size(C) ** 0.5),int(np.size(C) ** 0.5)))
    w,v = np.linalg.eigh(C_vectorized)
    eigenvectors = np.flip(v[:,-k:],1)
    eigenvalues = np.flip(w[-k:])
    return (eigenvectors,eigenvalues)


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for i in range(0, self.num_iterations):
            h = WeakClassifier(X=self.Xtrain, y=self.ytrain, weights=self.weights)
            h.train()
            h_j = h.predict(np.transpose(self.Xtrain))
            self.weakClassifiers = np.append(self.weakClassifiers, h)
            ε_index = self.ytrain != h_j
            ε_sum = np.sum(self.weights[ε_index]) / np.sum(self.weights)
            α = (1.2) * np.log((1. - ε_sum) / ε_sum)
            self.alphas = np.append(self.alphas, α)
            if ε_sum > self.eps:
                # self.weights[ε_index] = self.weights[ε_index] * np.exp(-α * h_j[ε_index] * self.ytrain[ε_index])
                self.weights[ε_index] *= np.exp(-α * h_j[ε_index] * self.ytrain[ε_index])
            else:
                break


    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        correct = np.sum(np.where(self.predict(self.Xtrain) == self.ytrain,1,0))
        incorrect = np.sum(np.where(self.predict(self.Xtrain) != self.ytrain,1,0))
        return ((correct, incorrect))


    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predict = [[h.predict(np.transpose(X))] for h in self.weakClassifiers]
        for i in range(0, len(self.alphas)):
            predict[i] = self.alphas[i] * np.array(predict[i])
        predict = np.sum(predict, axis=0)
        predict.reshape(np.size(predict))
        return np.sign(predict)

class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        feature_img = np.zeros(shape)
        y, x = self.position
        h, w = self.size
        feature_img[y:y+h//2,x:x+w] = 255
        feature_img[y:y + int(h/2), x:x + w] = 255
        feature_img[y+h//2:y+h,x:x+w] = 126
        return feature_img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        feature_img = np.zeros(shape)
        y, x = self.position
        h, w = self.size
        feature_img[y:y+h,x:x+w//2] = 255
        feature_img[y:y+h,x+w//2:x+w] = 126
        return feature_img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        feature_img = np.zeros(shape)
        y, x = self.position
        h, w = self.size
        feature_img[y:y+h, x:x+w] = 255
        feature_img[y+h//3:y+2*(h//3), x:x+w] = 126
        return feature_img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        feature_img = np.zeros(shape)
        y, x = self.position
        h, w = self.size
        feature_img[y:y+h, x:x+w] = 255
        feature_img[y:y+h, x+w//3:x+2*(w//3)] = 126
        return feature_img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        feature_img = np.zeros(shape)
        y, x = self.position
        h, w = self.size
        feature_img[y:y+h, x:x+w] = 126
        feature_img[y:y+h//2,x+w//2:x+w]=255
        feature_img[y+h//2:y+h,x:x+w//2]=255
        return feature_img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output_images/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output_images/{}.png".format(filename), X)
        print()
        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        y = y_1 = self.position[0]
        x = x_1 = self.position[1]
        h, w = self.size
        ii = ii.astype(float)
        if self.feat_type == (2, 1):
            # integral image method
            y_2 = y + h//2 - 1
            x_2 = x_1 + w - 1
            A = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y + h//2
            x_1 = x
            y_2 = y + h - 1
            x_2 = x_1 + w - 1
            B = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            return A - B


        if self.feat_type == (1, 2):
            y_2 = y_1 + h - 1
            x_2 = x_1 + w//2 - 1
            A = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y
            x_1 = x + w//2
            y_2 = y_1 + h - 1
            x_2 = x + w - 1
            B = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            return A - B

        if self.feat_type == (3, 1):
            y_2 = y + h//3 - 1
            x_2 = x_1 + w - 1
            A = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] +  ii[y_2, x_2]

            y_1 = y + h//3
            x_1 = x
            y_2 = y + h//3 + h//3 - 1
            x_2 = x_1 + w - 1
            B = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y + h//3 + h//3
            x_1 = x
            y_2 = y + h - 1
            x_2 = x_1 + w - 1
            C = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            return A - B + C

        if self.feat_type == (1, 3):
            y_2 = y_1 + h - 1
            x_2 = x_1 + w//3 - 1
            A = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y
            x_1 = x + w//3
            y_2 = y_1 + h - 1
            x_2 = x + w//3 + w//3 - 1
            B = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y
            x_1 = x + w//3 + w//3
            y_2 = y + h - 1
            x_2 = x + w - 1
            C = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            return A - B + C

        if self.feat_type == (2, 2):
            y_2 = y_1 + h//2 - 1
            x_2 = x_1 + w//2 - 1
            A = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y
            x_1 = x + w//2
            y_2 = y_1 + h//2 - 1
            x_2 = x + w - 1
            B = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y + h//2
            x_1 = x
            y_2 = y + h - 1
            x_2 = x_1 + w//2 - 1
            C = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

            y_1 = y + h//2
            x_1 = x + w//2
            y_2 = y + h - 1
            x_2 = x + w - 1
            D = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]
            return -A + B + C - D
        return None


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_img = [np.cumsum(np.cumsum(img, axis=0), axis=1) for img in images]
    return integral_img


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))
        self.threshold = 0.49

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_train(self):
        """ This function initializes self.scores, self.weights

        Args:
            None

        Returns:
            None
        """
    
        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        if not self.integralImages or not self.haarFeatures:
            print("No images provided. run convertImagesToIntegralImages() first")
            print("       Or no features provided. run creatHaarFeatures() first")
            return

        self.scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        # print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            self.scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        self.weights = np.hstack((weights_pos, weights_neg))

    def train(self, num_classifiers):
        """ Initialize and train Viola Jones face detector

        The function should modify self.weights, self.classifiers, self.alphas, and self.threshold

        Args:
            None

        Returns:
            None
        """
        self.init_train()
        # print(" -- select classifiers --")

        for i in range(num_classifiers):
            self.weights /= np.sum(self.weights)
            vj = VJ_Classifier(self.scores, self.labels, self.weights)
            vj.train()
            self.classifiers.append(vj)
            β = vj.error / (1.0 - vj.error)
            for i, (score, label) in enumerate(zip(self.scores, self.labels)):
                if vj.predict(score) == label:
                    self.weights[i] *= np.power(β, 1 - 0)
                else:
                    self.weights[i] *= np.power(β, 1 - 1)
            self.alphas.append(np.log(1.0 / β))


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))
        self.scores = scores

        for i in range(len(self.classifiers)):
            idx = self.classifiers[i].feature
            for j in range(0, len(ii)):
                self.scores[j, idx] = self.haarFeatures[idx].evaluate(ii[j])

        # strong classifier H(x)
        H = np.zeros((len(ii), len(self.classifiers)))

        for i in range(len(self.classifiers)):
            H[:, i] = [self.classifiers[i].predict(self.scores[j]) * self.alphas[i] for j in range(0, len(ii))]

        return [1 if np.sum(x) >= self.threshold * np.sum(self.alphas) else -1 for x in H]

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        window_size = 24
        # window_size_tuple = (24,24)
        windows = []
        top_left = []
        for y in range(0, img_gray.shape[0] - window_size + 1, 1):
            for x in range(0, img_gray.shape[1] - window_size + 1, 1):
                top_left.append([x, y])
                windows.append(np.array(img_gray[y:y + window_size, x:x + window_size]))

        # windows = sliding_window_view(img_gray,window_size_tuple)
        # windows_flatten = windows.reshape((windows.shape[0]*windows.shape[1],windows.shape[2],windows.shape[3])).tolist()
        # predictions = self.predict(windows_flatten)

        predictions = self.predict(windows)

        top_left_avg = (np.array(top_left)[np.array(predictions) == 1]).mean(axis=0).astype(np.int)
        cv2.rectangle(image,  top_left_avg ,  top_left_avg + window_size , (255, 0, 0), 1)
        cv2.imwrite("./output_images/"+ filename + '.png', image)

class CascadeClassifier:
    """Viola Jones Cascade Classifier Face Detection Method

    Lesson: 8C-L2, Boosting and face detection

    Args:
        f_max (float): maximum acceptable false positive rate per layer
        d_min (float): minimum acceptable detection rate per layer
        f_target (float): overall target false positive rate
        pos (list): List of positive images.
        neg (list): List of negative images.

    Attributes:
        f_target: overall false positive rate
        classifiers (list): Adaboost classifiers
        train_pos (list of numpy arrays):  
        train_neg (list of numpy arrays): 

    """
    def __init__(self, pos, neg, f_max_rate=0.30, d_min_rate=0.70, f_target = 0.07):
        
        train_percentage = 0.85

        pos_indices = np.random.permutation(len(pos)).tolist()
        neg_indices = np.random.permutation(len(neg)).tolist()

        train_pos_num = int(train_percentage * len(pos))
        train_neg_num = int(train_percentage * len(neg))

        pos_train_indices = pos_indices[:train_pos_num]
        pos_validate_indices = pos_indices[train_pos_num:]

        neg_train_indices = neg_indices[:train_neg_num]
        neg_validate_indices = neg_indices[train_neg_num:]

        self.train_pos = [pos[i] for i in pos_train_indices]
        self.train_neg = [neg[i] for i in neg_train_indices]

        self.validate_pos = [pos[i] for i in pos_validate_indices]
        self.validate_neg = [neg[i] for i in neg_validate_indices]

        self.f_max_rate = f_max_rate
        self.d_min_rate = d_min_rate
        self.f_target = f_target
        self.classifiers = []

    def predict(self, classifiers, img):
        """Predict face in a single image given a list of cascaded classifiers

        Args:
            classifiers (list of element type ViolaJones): list of ViolaJones classifiers to predict 
                where index i is the i'th consecutive ViolaJones classifier
            img (numpy.array): Input image

        Returns:
            Return 1 (face detected) or -1 (no face detected) 
        """

        # TODO
        raise NotImplementedError

    def evaluate_classifiers(self, pos, neg, classifiers):
        """ 
        Given a set of classifiers and positive and negative set
        return false positive rate and detection rate 

        Args:
            pos (list): Input image.
            neg (list): Output image file name.
            classifiers (list):  

        Returns:
            f (float): false positive rate
            d (float): detection rate
            false_positives (list): list of false positive images
        """

        # TODO
        raise NotImplementedError

    def train(self):
        """ 
        Trains a cascaded face detector

        Sets self.classifiers (list): List of ViolaJones classifiers where index i is the i'th consecutive ViolaJones classifier

        Args:
            None

        Returns:
            None
             
        """
        # TODO
        raise NotImplementedError


    def faceDetection(self, image, filename="ps6-5-b-1.jpg"):
        """Scans for faces in a given image using the Cascaded Classifier.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        raise NotImplementedError


