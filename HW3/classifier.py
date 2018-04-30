import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def train_classifier(features, labels, args):
    if args.classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=args.knn_k)
        clf.fit(features, labels)
        return clf
    elif args.classifier == 'svm':
        clf = SVC(C=args.svm_lambda)
        clf.fit(features, labels)
        return clf
    else:
        raise NotImplementedError()


def make_predictions(clf, features):
    prediction_probs = clf.predict_proba(features)
    return np.argmax(prediction_probs, axis=-1), \
        np.max(prediction_probs, axis=-1)


def show_confusion(labels, predictions):
    conf_mat = confusion_matrix(
        y_true=labels,
        y_pred=predictions,
        labels=['Cats', 'Dogs'])
    print(conf_mat)
