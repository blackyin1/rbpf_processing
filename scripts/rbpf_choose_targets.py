#!/usr/bin/python

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg

class Picker(object):

    def __init__(self):
        self.index = 0
        self.indices = []
        self.done = False

    def step(self):
        self.index += 1

    def set_target(self, event):
        self.indices.append(self.index)
        print "Setting target"

    def set_done(self, event):
        self.done = True
        plt.close()
        print "Done"

    def next(self, event):
        print "Next"
        plt.close()

def choose_targets(data_path):

    objects_dict = np.load(os.path.join(data_path, "data_summary.npz"))
    home_path = os.path.join(os.path.expanduser("~"), ".ros")
    observations_file = os.path.join(home_path, "detection_observations.npz")
    target_observations_file = os.path.join(home_path, "target_detection_observations.npz")

    observations_dict = np.load(observations_file)

    images = objects_dict['central_images']

    timestamps = observations_dict['timesteps']

    picker = Picker()

    for i in range(0, len(timestamps)):

        print images[i]

        img = mpimg.imread(images[i])

        fig, ax = plt.subplots()

        plt.imshow(img)

        plt.subplots_adjust(bottom=0.2)

        nxprev = plt.axes([0.59, 0.05, 0.1, 0.075])
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        nnext = Button(nxprev, 'Next')
        nnext.on_clicked(picker.next)
        bnext = Button(axnext, 'Set target')
        bnext.on_clicked(picker.set_target)
        bprev = Button(axprev, 'Done')
        bprev.on_clicked(picker.set_done)

        plt.show()

        picker.step()
        if picker.done:
            break

    inds = np.array(picker.indices, dtype=int)

    spatial_measurements = observations_dict['spatial_measurements']
    feature_measurements = observations_dict['feature_measurements']
    timesteps = observations_dict['timesteps']
    spatial_positions = observations_dict['spatial_positions']
    target_ids = observations_dict['target_ids']
    observation_ids = observations_dict['observation_ids']
    clouds = observations_dict['clouds']
    detection_type = observations_dict['detection_type']
    going_backward = observations_dict['going_backward']

    print spatial_measurements.shape, feature_measurements.shape, timesteps.shape, spatial_positions.shape, \
          target_ids.shape, observation_ids.shape, clouds.shape, len(detection_type), len(going_backward)

    print clouds

    spatial_measurements = np.vstack((spatial_measurements[inds, :], spatial_measurements))
    feature_measurements = np.vstack((feature_measurements[inds, :], feature_measurements))
    timesteps = np.concatenate((0*timesteps[inds], timesteps+1))
    spatial_positions = np.vstack((spatial_positions[inds, :], spatial_positions))
    target_ids = np.concatenate((np.arange(0, len(inds), dtype=int), target_ids))
    observation_ids = np.arange(0, len(timesteps), dtype=int)
    clouds = np.concatenate((clouds[inds], clouds))
    detection_type = np.concatenate((detection_type[inds], detection_type))
    going_backward = np.concatenate((going_backward[inds], going_backward))

    print spatial_measurements.shape, feature_measurements.shape, timesteps.shape, spatial_positions.shape, \
          target_ids.shape, observation_ids.shape, clouds.shape, len(detection_type), len(going_backward)

    print picker.indices
    np.savez(target_observations_file, spatial_measurements = spatial_measurements,
                                       feature_measurements = feature_measurements,
                                       timesteps = timesteps,
                                       spatial_positions = spatial_positions,
                                       target_ids = target_ids,
                                       observation_ids = observation_ids,
                                       spatial_measurement_std = observations_dict['spatial_measurement_std'],
                                       feature_measurement_std = observations_dict['feature_measurement_std'],
                                       clouds = clouds,
                                       detection_type = detection_type,
                                       going_backward = going_backward)

    #print timestamps

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data"
    else:
        choose_targets(sys.argv[1])
