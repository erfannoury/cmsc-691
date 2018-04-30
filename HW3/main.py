import sys
import argparse


import numpy as np


from dataset import read_dataset
from features import compute_features
from normalize import normalize_features
from classifier import train_classifier, make_predictions, show_confusion


CLASSIFIERS = ['knn', 'svm']
FEATURES = ['tinyimage', 'bow-patches', 'bow-sift', 'gist']


def main(args):
    parser = argparse.ArgumentParser(
        description='Train and evaluate a model on the Cats vs. Dogs dataset')

    parser.add_argument('-d', '--dataset-dir', required=True, type=str,
                        help='Path to the dataset')
    parser.add_argument('-f', '--feature', required=True, choices=FEATURES,
                        help='Select which feature representation to use. '
                        'Choices are {' + ', '.join(FEATURES) + '}')
    parser.add_argument('-c', '--classifier', required=True,
                        choices=CLASSIFIERS,
                        help='Select which classifier to use. '
                        'Choices are {' + ', '.join(CLASSIFIERS) + '}')
    parser.add_argument('-k', '--knn-k', default=3, type=int,
                        help='Number of neighbors for kNN classifier')
    parser.add_argument('-l', '--svm-lambda', default=1.0, type=float,
                        help='Lambda paramter for SVM')
    parser.add_argument('--tinyimage-patchdim', default=16, type=int)
    parser.add_argument('--patches-dictionarysize', default=128, type=int)
    parser.add_argument('--patches-radius', default=8, type=float)
    parser.add_argument('--patches-stride', default=12, type=int)
    parser.add_argument('--sift-dictionarysize', default=128, type=int)
    parser.add_argument('--sift-binsize', default=8, type=int,
                        help='Size of the bin in terms of number of pixels in '
                        'the image. Recall that SIFT has 4x4=16 bins.')
    parser.add_argument('--sift-stride', default=12, type=int,
                        help='Spacing between succesive x (and y) coordinates '
                        'for sampling dense features.')

    args = parser.parse_args(args)

    imdb = read_dataset(args.dataset_dir)

    features = compute_features(imdb, args)

    if args.feature != 'tinyimage':
        features = normalize_features(features)

    print(f'Experiment setup: trainining set: train, test set: val')
    clf = train_classifier(
        features[imdb.train_indices, :],
        imdb.class_ids[imdb.train_indices],
        args)
    val_preds, val_scores = make_predictions(
        clf,
        features[imdb.val_indices, :])
    show_confusion(imdb.class_ids[imdb.val_indices], val_preds)

    print(f'Experiment setup: trainining set: train+val, test set: test')
    clf = train_classifier(
        features[np.hstack((imdb.train_indices, imdb.val_indices)), :],
        imdb.class_ids[np.hstack((imdb.train_indices, imdb.val_indices))],
        args)
    test_preds, test_scores = make_predictions(
        clf,
        features[imdb.test_indices, :])
    show_confusion(imdb.class_ids[imdb.test_indices], test_preds)


if __name__ == '__main__':
    main(sys.argv[1:])
