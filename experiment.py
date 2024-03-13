"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import numba
import time
import ps6

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

YALE_FACES_DIR = os.path.join(INPUT_DIR, 'Yalefaces')
FRUITS_DIR = os.path.join(INPUT_DIR, 'fruits')
POS_DIR = os.path.join(INPUT_DIR, "pos")
POS2_DIR = os.path.join(INPUT_DIR, "pos2")
NEG_DIR = os.path.join(INPUT_DIR, "neg")
NEG2_DIR = os.path.join(INPUT_DIR, "neg2")


def load_images_from_dir(data_dir, size=(24, 24), ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs

# Utility function
def plot_eigen_faces(eig_vecs, fig_name="", visualize=False):
    r = np.ceil(np.sqrt(len(eig_vecs)))
    c = int(np.ceil(len(eig_vecs)/r))
    r = int(r)
    fig = plt.figure()

    for i,v in enumerate(eig_vecs):
        sp = fig.add_subplot(r,c,i+1)

        plt.imshow(v.reshape(32,32).real, cmap='gray')
        sp.set_title('eigenface_%i'%i)
        sp.axis('off')

    fig.subplots_adjust(hspace=.5)

    if visualize:
        plt.show()

    if not fig_name == "":
        plt.savefig("{}".format(fig_name))


# Functions you need to complete
def visualize_mean_face(x_mean, size, new_dims):
    """Rearrange the data in the mean face to a 2D array

    - Organize the contents in the mean face vector to a 2D array.
    - Normalize this image.
    - Resize it to match the new dimensions parameter

    Args:
        x_mean (numpy.array): Mean face values.
        size (tuple): x_mean 2D dimensions
        new_dims (tuple): Output array dimensions

    Returns:
        numpy.array: Mean face uint8 2D array.
    """
    reshape = np.reshape(x_mean, size)
    resize = cv2.resize(reshape,new_dims)
    norm = cv2.normalize(resize, None, 0, 255, cv2.NORM_MINMAX)

    # cv2.imshow('reshape',reshape)
    # cv2.imshow('resize', resize)
    # cv2.imshow('norm', norm)
    # cv2.waitKey()
    return resize


def part_1a_1b():

    orig_size = (192, 231)
    small_size = (32, 32)
    X, y = ps6.load_images(YALE_FACES_DIR, small_size)

    # Get the mean face
    x_mean = ps6.get_mean_face(X)

    x_mean_image = visualize_mean_face(x_mean, small_size, orig_size)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ps6-1-a-1.png"), x_mean_image)

    # PCA dimension reduction
    k = 10
    eig_vecs, eig_vals = ps6.pca(X, k)

    plot_eigen_faces(eig_vecs.T, "ps6-1-b-1.png")


def part_1c():
    p = .3  # Select a split percentage value
    k = 7  # Select a value for k

    size = (32, 32)
    X, y = ps6.load_images(YALE_FACES_DIR, size)
    Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

    # training
    mu = ps6.get_mean_face(Xtrain)
    eig_vecs, eig_vals = ps6.pca(Xtrain, k)
    Xtrain_proj = np.dot(Xtrain - mu, eig_vecs)

    # testing
    mu = ps6.get_mean_face(Xtest)
    Xtest_proj = np.dot(Xtest - mu, eig_vecs)

    good = 0
    bad = 0

    for i, obs in enumerate(Xtest_proj):

        dist = [np.linalg.norm(obs - x) for x in Xtrain_proj]

        idx = np.argmin(dist)
        y_pred = ytrain[idx]

        if y_pred == ytest[i]:
            good += 1

        else:
            bad += 1

    print('Good predictions = ', good, 'Bad predictions = ', bad)
    print('{0:.2f}% accuracy'.format(100 * float(good) / (good + bad)))


def part_2a():
    y0 = 1
    y1 = 2

    X, y = ps6.load_images(FRUITS_DIR)

    # Select only the y0 and y1 classes
    idx = y == y0
    idx |= y == y1

    X = X[idx,:]
    # X = X[idx]
    y = y[idx]

    # Label them 1 and -1
    y0_ids = y == y0
    y1_ids = y == y1
    y[y0_ids] = 1
    y[y1_ids] = -1

    p = 0.3
    Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytrain)))
    # TODO: find which of these labels match ytrain and report its accuracy
    correct = np.where(ytrain == rand_y,1,0)
    rand_accuracy = 100 * np.sum(correct)/np.size(correct)

    print('(Random) Training accuracy: {0:.2f}%'.format(rand_accuracy))

    # Using Weak Classifier
    uniform_weights = np.ones((Xtrain.shape[0],)) / Xtrain.shape[0]
    wk_clf = ps6.WeakClassifier(Xtrain, ytrain, uniform_weights)
    wk_clf.train()
    wk_results = [wk_clf.predict(x) for x in Xtrain]
    # TODO: find which of these labels match ytrain and report its accuracy

    correct = np.where(wk_results == ytrain,1,0)
    wk_accuracy = 100 * np.sum(correct)/np.size(correct)

    print('(Weak) Training accuracy {0:.2f}%'.format(wk_accuracy))

    num_iter = 5

    boost = ps6.Boosting(Xtrain, ytrain, num_iter)
    boost.train()
    good, bad = boost.evaluate()
    boost_accuracy = 100 * float(good) / (good + bad)
    print('(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy))

    # Picking random numbers
    rand_y = np.random.choice([-1, 1], (len(ytest)))
    # TODO: find which of these labels match ytest and report its accuracy
    correct = np.where(ytest == rand_y,1,0)
    rand_accuracy = 100 * np.sum(correct)/np.size(correct)
    print('(Random) Testing accuracy: {0:.2f}%'.format(rand_accuracy))

    # Using Weak Classifier
    wk_results = [wk_clf.predict(x) for x in Xtest]
    # TODO: find which of these labels match ytest and report its accuracy
    correct = np.where(wk_results == ytest,1,0)
    wk_accuracy = 100 * np.sum(correct)/np.size(correct)
    print('(Weak) Testing accuracy {0:.2f}%'.format(wk_accuracy))

    y_pred = boost.predict(Xtest)
    # TODO: find which of these labels match ytest and report its accuracy
    correct = np.where(y_pred == ytest,1,0)
    boost_accuracy = 100 * np.sum(correct)/np.size(correct)
    print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy))


def part_3a():
    """Complete the remaining parts of this section as instructed in the
    instructions document."""

    feature1 = ps6.HaarFeature((2, 1), (25, 30), (50, 100))
    feature1.preview((200, 200), filename="ps6-3-a-1")

    feature1 = ps6.HaarFeature((1, 2), (10, 25), (50, 150))
    feature1.preview((200, 200), filename="ps6-3-a-2")

    feature1 = ps6.HaarFeature((3, 1), (50, 50), (100, 50))
    feature1.preview((200, 200), filename="ps6-3-a-3")

    feature1 = ps6.HaarFeature((1, 3), (50, 125), (100, 50))
    feature1.preview((200, 200), filename="ps6-3-a-4")

    feature1 = ps6.HaarFeature((2, 2), (50, 25), (100, 150))
    feature1.preview((200, 200), filename="ps6-3-a-5")

    # TODO: Generate and save all required images
    return None


def part_4_a_b():

    pos = load_images_from_dir(POS_DIR)
    neg = load_images_from_dir(NEG_DIR)

    train_pos = pos
    train_neg = neg
    images = train_pos + train_neg
    labels = np.array(len(train_pos) * [1] + len(train_neg) * [-1])

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(train_pos, train_neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(5)

    # VJ.haarFeatures[VJ.classifiers[0].feature].preview(filename="ps6-4-b-1")
    VJ.haarFeatures[VJ.classifiers[1].feature].preview(filename="ps6-4-b-2")

    predictions = VJ.predict(images)
    correct = np.where(predictions == labels,1,0)
    vj_accuracy = 100.0 * (correct.sum()) / len(labels)

    print("Prediction accuracy on training: {0:.2f}%".format(vj_accuracy))

    pos2 = load_images_from_dir(POS2_DIR)
    neg2 = load_images_from_dir(NEG2_DIR)

    test_pos = pos2
    test_neg = neg2
    test_images = test_pos + test_neg
    real_labels = np.array(len(test_pos) * [1] + len(test_neg) * [-1])
    predictions = VJ.predict(test_images)
    correct = np.where(predictions == real_labels, 1, 0)
    vj_accuracy = 100.0 * (correct.sum()) / len(real_labels)

    print("Prediction accuracy on testing: {0:.2f}%".format(vj_accuracy))


def part_4_c():
    pos = load_images_from_dir(POS_DIR)[:34]
    neg = load_images_from_dir(NEG_DIR)

    images = pos + neg

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(pos, neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(4)

    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    image = cv2.resize(image, (120, 60))
    VJ.faceDetection(image, filename="ps4-4-c-1")

def part_5_b():
    pos2 = load_images_from_dir(POS2_DIR)
    neg2 = load_images_from_dir(NEG2_DIR)

    CC = ps6.CascadeClassifier(pos2, neg2)
    CC.train()

    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    image = cv2.resize(image, (120, 60))
    CC.faceDetection(image, filename="cascade")


if __name__ == "__main__":
    part_1a_1b()
    part_1c()
    part_2a()
    part_3a()
    part_4_a_b()
    part_4_c()



    # Comparing Integral image to np.sum approach by
    # calculating the sum of all pixels in this range [10:10, 50:50]


    def np_sum(img_gray):
        section = img_gray[0:100,0:100]
        for i in range(1000):
            np.sum(section)
        return


    def convert_image_to_integral_images(image):
        return np.cumsum(np.cumsum(image, axis=0), axis=1).astype(np.int32)

    img = cv2.imread('.\input_images\man.jpeg',-1)
    # --numpy sum approach
    start = time.time()
    A = np_sum(img)
    np_end = time.time()
    # II sum approach
    ii = convert_image_to_integral_images(img)
    x = y = x_1 = y_1 = 0
    h = w = 100
    ii = ii.astype(float)
    for i in range(1000):
        y_2 = y + h // 2 - 1
        x_2 = x_1 + w - 1
        A = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

        y_1 = y + h // 2
        x_1 = x
        y_2 = y + h - 1
        x_2 = x_1 + w - 1
        B = ii[y_1 - 1, x_1 - 1] - ii[y_1 - 1, x_2] - ii[y_2, x_1 - 1] + ii[y_2, x_2]

    result = A - B

    ii_end = time.time()
    print('np.sum() runtime', np_end - start)
    print('Integral Image runtime', ii_end - np_end )