from __future__ import division
import numpy as np


class LogisticRegression:
    def __init__(self, data, labels, alpha=1, num_iters=100, regularized=False, debug=False, normalization='l2'):

        self.normalization_mode = normalization
        self.regularized = regularized
        self.debug = debug
        self.num_iters = num_iters
        self.alpha = alpha
        assert (len(np.unique(labels)) >= 2)
        pass

    def train(self, data, old_labels, unique_classes):
        """
        Train logistic regression model
        :param data: training data
        :param old_labels: original labels
        :param unique_classes: number of classes
        :return: model parameters
        """

        print('training....')

        m, n = data.shape

        labels = np.zeros(old_labels.shape)

        original_label_name = np.unique(old_labels)

        label_list = range(len(original_label_name))

        for each in zip(original_label_name, label_list):
            o_label_name = each[0]
            new_label_name = each[1]
            labels[np.where(old_labels == o_label_name)] = new_label_name

        labels = labels.reshape((len(labels), 1))

        num_classes = len(unique_classes)

        init_thetas = []  # to hold initial values of theta

        thetas = []  # to hold final values of theta to return

        if num_classes > 2:
            for eachInitTheta in range(num_classes):
                theta_init = np.zeros((n, 1))
                init_thetas.append(theta_init)
            for eachClass in range(num_classes):
                local_labels = np.zeros(labels.shape)

                local_labels[np.where(labels == eachClass)] = 1

                assert (len(np.unique(local_labels)) == 2)
                assert (len(local_labels) == len(labels))

                init_theta = init_thetas[eachClass]
                
                new_theta = self.compute_gradient(data, local_labels, init_theta)

                thetas.append(new_theta)

        return thetas

    def classify(self, data, thetas):
        """
        Predict label
        :param data:
        :param thetas:
        :return:
        """
        assert (len(thetas) > 0)

        if len(thetas) > 1:
            mvals = []
            for eachTheta in thetas:
                mvals.append(self.sigmoid(np.dot(data, eachTheta)))
            return mvals.index(max(mvals)) + 1

        elif len(thetas) == 1:
            cval = round(self.sigmoid(np.dot(data, thetas[0]))) + 1.0
            return cval

    def sigmoid(self, data):
        data = np.array(data, dtype=np.longdouble)
        g = 1 / (1 + np.exp(-data))
        return g

    def compute_gradient(self, data, labels, init_theta):
        """
        computer gradient
        :param data:
        :param labels:
        :param init_theta:
        :return:
        """
        alpha = self.alpha
        num_iters = self.num_iters
        m, n = data.shape
        regularized = self.regularized
        theta = init_theta
        for eachIteration in range(num_iters):
            bias = self.sigmoid(np.dot(data, init_theta) - labels)
            x = (1 / m) * np.transpose(data)
            grad = np.dot(x, bias)
            
            bias = self.sigmoid(np.dot(data, init_theta)) - labels
            for i in range(len(grad)):
                xj = (data[:, i].reshape((data[:, i].shape[0], 1)))
                if regularized:
                    grad[i] = (np.sum(bias * xj) + init_theta[i]) / m
                else:
                    grad[i] = np.sum(bias * xj) / m
            # update gradient
            theta = theta - (np.dot((alpha / m), grad))
        return theta
