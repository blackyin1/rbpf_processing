#!/usr/bin/python

import os
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import sys
import json

# This code comes from https://code.oursky.com/tensorflow-svm-image-classifications-engine/

def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.

    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(image_paths, verbose=False):
    """
    extract_features computed the inception bottleneck feature for a list of images

    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))

    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for i, image_path in enumerate(image_paths):
            if verbose:
                print('Processing %s...' % (image_path))

            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image_path, 'rb').read()
            feature = sess.run(flattened_tensor, {
                'DecodeJpeg/contents:0': image_data
            })
            features[i, :] = np.squeeze(feature)

    return features

def load_and_extract(model_path, data_path):

    with open(os.path.join(data_path, "feature_rgb_images.json")) as data_file:
        feature_images_dict = json.load(data_file)

    images = feature_images_dict['value0']

    print images
    print "Extracting features"

    create_graph(model_path)
    features = extract_features(images, verbose=True)

    np.savez(os.path.join(data_path, "deep_rgb_features.npz"), features = features)

    print "Features: ", features.shape

if __name__ == '__main__':

    model_path = "/home/nbore/instance_places/catkin_ws/src/rbpf_processing/inception/tensorflow_inception_graph.pb"

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data"
    else:
        load_and_extract(model_path, sys.argv[1])
