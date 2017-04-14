#!/usr/bin/python

import numpy as np
import os
import sys

def replace_clouds_images(observations_file, data_path):

    objects_file = np.load(os.path.join(data_path, "data_summary.npz"))
    home_path = os.path.join(os.path.expanduser("~"), ".ros")

    observations_dict = np.load(observations_file)

    spatial_measurements = observations_dict['spatial_measurements']
    timesteps = observations_dict['timesteps']

    clouds = objects_file['clouds']
    central_images = objects_file['central_images']

    init_inds = np.where(timesteps == 0)[0]
    nbr_targets = len(init_inds)

    print nbr_targets
    print init_inds

    detection_poses = np.reshape(spatial_measurements[nbr_targets:], (-1, 3))
    inds = np.zeros((nbr_targets,), dtype=int)

    for i in init_inds:
        j = np.argmin(np.linalg.norm(detection_poses - spatial_measurements[i].flatten(), axis=1))
        inds[i] = j
        print "Target: ", i, ", corresponding measurement: ", j

    clouds = np.concatenate((clouds[inds], clouds))
    central_images = np.concatenate((central_images[inds], central_images))

    print clouds.shape
    print observations_dict['clouds'].shape

    print central_images.shape
    print observations_dict['central_images'].shape

    #print central_images

    np.savez(observations_file, spatial_measurements = observations_dict['spatial_measurements'],
                                feature_measurements = observations_dict['feature_measurements'],
                                timesteps = observations_dict['timesteps'],
                                spatial_positions = observations_dict['spatial_positions'],
                                target_ids = observations_dict['target_ids'],
                                observation_ids = observations_dict['observation_ids'],
                                spatial_measurement_std = observations_dict['spatial_measurement_std'],
                                feature_measurement_std = observations_dict['feature_measurement_std'],
                                clouds = clouds,
                                central_images = central_images,
                                detection_type = observations_dict['detection_type'],
                                going_backward = observations_dict['going_backward'],
                                location_ids = observations_dict['location_ids'],
                                dims = observations_dict['dims'],
                                measurement_covariance = observations_dict['measurement_covariance'])

    #print timestamps

if __name__ == '__main__':

    if len(sys.argv) == 3:
        replace_clouds_images(sys.argv[1], sys.argv[2])
    else:
        print "Usage: ", sys.argv[0], " path/to/observations.npz path/to/data"
