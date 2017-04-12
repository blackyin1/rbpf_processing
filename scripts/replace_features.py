#!/usr/bin/python

import numpy as np
import os
import sys

def compute_new_features(measurement_covariance, features, dims):

    vols = np.reshape(np.prod(dims, axis=1), (-1, 1))
    new_features = np.hstack((features, vols))

    print vols

    new_shape = measurement_covariance.shape[0]+1
    new_covariance = np.zeros((new_shape, new_shape))
    new_covariance[:-1, :-1] = measurement_covariance
    new_covariance[-1, -1] = 0.01

    print new_features.shape
    print new_covariance

    #return new_covariance, new_features

    return measurement_covariance, features

def replace_features(observations_file, data_path):

    objects_file = np.load(os.path.join(data_path, "data_summary.npz"))
    home_path = os.path.join(os.path.expanduser("~"), ".ros")

    observations_dict = np.load(observations_file)

    features_file = np.load(os.path.join(data_path, "reduced_object_features.npz"))
    features = features_file['features']
    measurement_covariance = features_file['measurement_covariance']


    spatial_measurements = observations_dict['spatial_measurements']
    timesteps = observations_dict['timesteps']
    dims = observations_dict['dims']

    init_inds = np.where(timesteps == 0)[0]
    nbr_targets = len(init_inds)

    print nbr_targets
    print init_inds

    #temp = spatial_measurements[nbr_targets:].flatten()
    #print spatial_measurements[nbr_targets:]
    detection_poses = np.reshape(spatial_measurements[nbr_targets:], (-1, 3))
    #print detection_poses

    inds = np.zeros((nbr_targets,), dtype=int)

    for i in init_inds:
        j = np.argmin(np.linalg.norm(detection_poses - spatial_measurements[i].flatten(), axis=1))
        inds[i] = j
        print "Target: ", i, ", corresponding measurement: ", j

    new_covariance, new_features = compute_new_features(measurement_covariance, features, dims)

    new_features = np.vstack((new_features[inds], new_features))

    print new_features.shape
    print features.shape
    print observations_dict['feature_measurements'].shape

    np.savez(observations_file, spatial_measurements = observations_dict['spatial_measurements'],
                                feature_measurements = new_features,
                                timesteps = observations_dict['timesteps'],
                                spatial_positions = observations_dict['spatial_positions'],
                                target_ids = observations_dict['target_ids'],
                                observation_ids = observations_dict['observation_ids'],
                                spatial_measurement_std = observations_dict['spatial_measurement_std'],
                                feature_measurement_std = observations_dict['feature_measurement_std'],
                                clouds = observations_dict['clouds'],
                                central_images = observations_dict['central_images'],
                                detection_type = observations_dict['detection_type'],
                                going_backward = observations_dict['going_backward'],
                                location_ids = observations_dict['location_ids'],
                                dims = observations_dict['dims'],
                                measurement_covariance = new_covariance)

    #print timestamps

if __name__ == '__main__':

    if len(sys.argv) == 3:
        replace_features(sys.argv[1], sys.argv[2])
    else:
        print "Usage: ", sys.argv[0], " path/to/observations.npz path/to/data"
