#!/usr/bin/python

import numpy as np
import os
import sys

def convert_observations(data_path):

    features_file = np.load(os.path.join(data_path, "reduced_object_features.npz"))
    objects_file = np.load(os.path.join(data_path, "data_summary.npz"))
    home_path = os.path.join(os.path.expanduser("~"), ".ros")
    observations_file = os.path.join(home_path, "detection_observations.npz")

    timestamps = objects_file['timestamps']

    initialization_ids = -1*np.ones((len(timestamps),), dtype=int)
    initialization_ids[:4] = np.arange(0, 4, dtype=int)

    spatial_positions = np.zeros((len(timestamps), 4, 2))

    clouds = objects_file['clouds']

    #for i in range(0, 100):
    #    print i, timestamps[i], clouds[i]

    np.savez(observations_file, spatial_measurements = objects_file['poses'],
                                feature_measurements = features_file['features'],
                                timesteps = timestamps,
                                spatial_positions = spatial_positions,
                                target_ids = initialization_ids,
                                observation_ids = np.arange(0, len(timestamps), dtype=int),
                                spatial_measurement_std = 0.3,
                                feature_measurement_std = 5.0,
                                clouds = clouds)

    #print timestamps

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data"
    else:
        convert_observations(sys.argv[1])
