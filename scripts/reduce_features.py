#!/usr/bin/python

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import sys
import os
import json
import rospkg

def dimension_reduction(data_path):

    #with open(os.path.join(data_path, "feature_labels.json")) as data_file:
    #    feature_labels_dict = json.load(data_file)

    #labels = feature_labels_dict['value0']

    rospack = rospkg.RosPack()
    pkg_path = os.path.abspath(rospack.get_path("rbpf_processing"))
    prior_path = os.path.join(pkg_path, "data", "prior_full_features.npz")
    priorfile = np.load(prior_path)
    prior_features = priorfile['features']

    npzfile = np.load(os.path.join(data_path, "deep_object_features.npz"))
    features = npzfile['features']

    print "Computing tsne reduction..."
    N = features.shape[0]

    all_features = np.vstack((features, prior_features))

    n_components = 2
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    red_all_features = tsne.fit_transform(all_features)
    red_features = red_all_features[:N]

    print "Feature shape: ", features.shape
    print "Red all feature shape: ", red_all_features.shape
    print "Reduced shape: ", red_features.shape
    #print "Labels length: ", len(labels)

    print "Done computing tsne reduction..."

    np.savez(os.path.join(data_path, "reduced_object_features.npz"), features=red_features)

    print "Done saving..."

def plot_reduction(data_path):

    npzfile = np.load(os.path.join(data_path, "reduced_object_features.npz"))
    features = npzfile['features']

    plt.scatter(features[:, 0], features[:, 1], marker='o', cmap=plt.cm.Spectral, s=70)

    plt.show()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data (--plot)"
    elif len(sys.argv) == 2:
        dimension_reduction(sys.argv[1])
    elif sys.argv[2] == "--plot":
        plot_reduction(sys.argv[1])
    else:
        print "Usage: ", sys.argv[0], " path/to/data (--plot)"
