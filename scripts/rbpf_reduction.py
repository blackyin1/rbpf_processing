#!/usr/bin/python

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import sys
import os
import json

def dimension_reduction(data_path):

    with open(os.path.join(data_path, "feature_labels.json")) as data_file:
        feature_labels_dict = json.load(data_file)

    labels = feature_labels_dict['value0']

    npzfile = np.load(os.path.join(data_path, "deep_features.npz"))
    features = npzfile['features']

    print "Computing tsne reduction..."

    n_components = 2
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    red_features = tsne.fit_transform(features)

    print "Feature shape: ", features.shape
    print "Reduced shape: ", red_features.shape
    print "Labels length: ", len(labels)

    print "Done computing tsne reduction..."

    np.savez(os.path.join(data_path, "reduced_deep_features.npz"), features=red_features)

    print "Done saving..."

def plot_reduction_class(data_path, labels):

    label_dict = {}

    label_counter = 0
    for label in labels:
        if label not in label_dict:
            label_dict[label] = label_counter
            label_counter += 1
    print label_dict

    npzfile = np.load(os.path.join(data_path, "reduced_deep_features.npz"))
    features = npzfile['features']

    marker_primitives = ['o', 'v', '>', '<', '^', '*', '+', 'x', 'D', 'h', 'H']
    hsv = plt.get_cmap('hsv')
    color_primitives = hsv(np.linspace(0, 1.0, len(label_dict)))
    N = len(marker_primitives)

    colors = [label_dict[label] for label in labels]
    markers = [marker_primitives[i % N] for i in range(0, len(label_dict))]

    handles = []

    #np.array([ind for ind, key, val in ], dtype=int)
    for i, (key, val) in enumerate(label_dict.items()):
        inds = np.array([ind for ind, label in enumerate(labels) if label == key], dtype=int)
        handles.append(plt.scatter(features[inds, 0], features[inds, 1], c=color_primitives[i], marker=markers[i], cmap=plt.cm.Spectral, s=100))

    handle_labels = [key for key, val in label_dict.items()]
    plt.legend(handles, handle_labels, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)

    #for i in range(0, features.shape[0]):
    #    plt.scatter(features[i, 0], features[i, 1], c=color_primitives[i], marker=markers[i], cmap=plt.cm.Spectral, s=40)
    #plt.scatter(features[:, 0], features[:, 1], c=colors, marker='o', s=40)
    plt.show()

def plot_reduction(data_path):

    with open(os.path.join(data_path, "feature_labels.json")) as data_file:
        feature_labels_dict = json.load(data_file)

    labels = [label.replace(" ", "") for label in feature_labels_dict['value0']]
    #print labels

    plot_reduction_class(data_path, labels)

    labels = [label.rstrip('1234567890') for label in labels]

    plot_reduction_class(data_path, labels)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data (--plot)"
    elif len(sys.argv) == 2:
        dimension_reduction(sys.argv[1])
    elif sys.argv[2] == "--plot":
        plot_reduction(sys.argv[1])
    else:
        print "Usage: ", sys.argv[0], " path/to/data (--plot)"
