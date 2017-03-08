#!/usr/bin/python

import numpy as np
import rospy
from rbpf_mtt.msg import ObjectMeasurement, GMMPoses
from geometry_msgs.msg import PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty, Int32
from std_srvs.srv import Empty as EmptySrv
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.25)

    ax.add_artist(ellip)
    return ellip

class FeatureVisNode(object):

    def __init__(self):

        print "Initializing visualizer..."

        self.data_path = rospy.get_param('~data_path')

        npzfile = np.load(os.path.join(self.data_path, "reduced_object_features.npz"))
        self.features = npzfile['features']
        self.feature_estimates = None
        self.feature_covs = None
        self.last_time = -1
        self.measurements = []

        rospy.Subscriber("feature_estimates", GMMPoses, self.estimate_callback)
        rospy.Subscriber("sim_filter_measurements", ObjectMeasurement, self.measurement_callback)

        print "Done initializing..."

    def save_feature_image(self):

        print "Saving feature figure!"

        if self.feature_estimates is None:
            return

        feature_measurements = np.array(self.measurements)
        nbr_targets = self.feature_estimates.shape[0]

        for j in range(0, nbr_targets):
            #cov = 20.*np.identity(2)
            cov = self.feature_covs[j]
            plot_cov_ellipse(cov, self.feature_estimates[j])

        plt.scatter(self.features[:, 0], self.features[:, 1], marker='o', c='blue', cmap=plt.cm.Spectral, s=70)
        plt.scatter(self.feature_estimates[:, 0], self.feature_estimates[:, 1], marker='o', c='red', cmap=plt.cm.Spectral, s=70)
        if len(self.measurements) > 0:
            plt.scatter(feature_measurements[:, 0], feature_measurements[:, 1], marker='o', c='yellow', cmap=plt.cm.Spectral, s=70)

        plt.savefig("features.png")
        plt.clf()
        plt.cla()

    def estimate_callback(self, feats):

        print "Got estimate callback!"

        nbr_targets = len(feats.modes)
        self.feature_estimates = np.zeros((nbr_targets, 2))
        self.feature_covs = np.zeros((nbr_targets, 2, 2))

        for j in range(0, nbr_targets):
            self.feature_estimates[j, 0] = feats.modes[j].pose.position.x
            self.feature_estimates[j, 1] = feats.modes[j].pose.position.y
            for y in range(0, 2):
                for x in range(0, 2):
                    self.feature_covs[j, y, x] = feats.modes[j].covariance[y*6+x]

        self.save_feature_image()

    def measurement_callback(self, meas):

        print "Got measurement callback!"

        if meas.timestep != self.last_time:
            self.measurements = []
            self.last_time = meas.timestep

        self.measurements.append([meas.feature[0], meas.feature[1]])



if __name__ == '__main__':

    rospy.init_node('visualize_features', anonymous=True)

    fvn = FeatureVisNode()

    rospy.spin()
