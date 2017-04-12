#!/usr/bin/python

import numpy as np
import os
import sys

def convert_observations(data_path, dummy_features):

    objects_file = np.load(os.path.join(data_path, "data_summary.npz"))
    home_path = os.path.join(os.path.expanduser("~"), ".ros")
    observations_file = os.path.join(home_path, "various_detection_observations.npz")

    timestamps = objects_file['timestamps']

    if dummy_features:
        features = np.ones((len(timestamps), 2)) + np.random.uniform(low=-.1, high=.1, size=(len(timestamps), 2))
        measurement_covariance = np.identity(2)
    else:
        features_file = np.load(os.path.join(data_path, "reduced_object_features.npz"))
        features = features_file['features']
        measurement_covariance = features_file['measurement_covariance']

    initialization_ids = -1*np.ones((len(timestamps),), dtype=int)
    initialization_ids[:4] = np.arange(0, 4, dtype=int)

    spatial_positions = np.zeros((len(timestamps), 4, 2))

    clouds = objects_file['clouds']
    central_images = objects_file['central_images']

    #for i in range(0, 100):
    #    print i, timestamps[i], clouds[i]

    np.savez(observations_file, spatial_measurements = objects_file['poses'],
                                feature_measurements = features,
                                timesteps = timestamps,
                                spatial_positions = spatial_positions,
                                target_ids = initialization_ids,
                                observation_ids = np.arange(0, len(timestamps), dtype=int),
                                spatial_measurement_std = 0.3,
                                feature_measurement_std = 5.0,
                                clouds = clouds,
                                central_images = central_images,
                                detection_type = objects_file['detection_type'],
                                going_backward = objects_file['going_backward'],
                                location_ids = objects_file['location_ids'],
                                dims = objects_file['dims'],
                                measurement_covariance = measurement_covariance)

    #print timestamps

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data (--dummy)"
    elif len(sys.argv) == 2:
        convert_observations(sys.argv[1], False)
    elif sys.argv[2] == "--dummy":
        convert_observations(sys.argv[1], True)
    else:
        print "Usage: ", sys.argv[0], " path/to/data (--dummy)"
