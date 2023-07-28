# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:59:15 2021

@author: PC-NEURON
"""

import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix


def NormalizeData(data):
    """Performs the data normalization"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


bae1_train = np.load('./data/bae1/bae1_tr.npz')
bae2_train = np.load('./data/bae2/bae2_tr.npz')
mvt_train = np.load('./data/mvt/mvt_tr.npz')

bae1_test = np.load('./data/bae1/bae1_ts.npz')
bae2_test = np.load('./data/bae2/bae2_ts.npz')
mvt_test = np.load('./data/mvt/mvt_ts.npz')


try:   
    train_labels = bae1_train['labels']
    test_labels = bae1_test['labels']
    
    bae1_train = bae1_train['ae_out']
    bae2_train = bae2_train['ae_out']
    mvt_train = mvt_train['ae_out']
    
    bae1_test = bae1_test['ae_out']
    bae2_test = bae2_test['ae_out']
    mvt_test = mvt_test['ae_out']
except:
    print('Loading data should be numpy array and has "ae_out" and "labels" keys.')
    sys.exit(1)

# fit the model
clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
clf.fit(bae1_train)
y_pred_train = clf.predict(bae1_train)
y_pred_test = clf.predict(bae1_test)

n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size


# Plot non-normalized and normalized   confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

class_names = ['OK', 'NOK']
# classifier = svm.OneClassSVM(nu = nus[int(np.argmax(roc_scores))], kernel = 'rbf', gamma = 'auto')

setattr(clf, "_estimator_type", "classifier")
#classifier.predict()

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, bae1_test, test_labels*2-1,
                                  display_labels=class_names,
                                  cmap=plt.cm.Blues,
                                  normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

# # plot the line, the points, and the nearest vectors to the plane
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.title("Novelty Detection")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

# s = 40
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
#                  edgecolors='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
#                 edgecolors='k')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([a.collections[0], b1, b2, c],
#            ["learned frontier", "training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left",
#            prop=matplotlib.font_manager.FontProperties(size=11))
# plt.xlabel(
#     "error train: %d/200 ; errors novel regular: %d/40 ; "
#     "errors novel abnormal: %d/40"
#     % (n_error_train, n_error_test, n_error_outliers))
# plt.show()