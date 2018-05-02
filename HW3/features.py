import numpy as np


def compute_features(imdb, args):
    if args.feature == 'tinyimage':
        return tinyimage_features(imdb, args.tinyimage_patchdim)
    elif args.feature == 'bow-patches':
        return bow_patch_features(
            imdb, args.patches_dictionarysize, args.patches_radius,
            args.patches_stride)
    elif args.feature == 'bow-sift':
        return bow_sift_features(
            imdb, args.sift_dictionarysize, args.sift_radius,
            args.sift_stride)
    else:
        raise NotImplementedError('Selected feature not yet implemented')


def tinyimage_features(imdb, patchdim):
    raise NotImplementedError()


def bow_patch_features(imdb, dictionarysize, radius, stride):
    raise NotImplementedError()


def bow_sift_features(imdb, dictionarysize, radius, stride):
    raise NotImplementedError()
