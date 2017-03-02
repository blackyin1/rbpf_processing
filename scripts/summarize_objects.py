#!/usr/bin/python

import os
import fnmatch
import sys
import re
import subprocess
import json
import numpy as np
#import natsort

def sortkey_natural(s):
    return tuple(int(part) if re.match(r'[0-9]+$', part) else part for part in re.split(r'([0-9]+)', s))

def get_sweep_xmls(data_path):

    file_list = []

    # Walk through directory
    for dname, sdname, files in os.walk(data_path):
        for filename in files:
            if fnmatch.fnmatch(filename, "room.xml"): # Match search string
                file_list.append(os.path.join(dname, filename))
    file_list = sorted(file_list, key=sortkey_natural)

    return file_list

def get_objects(s):

    dir_list = []

    sweep_path = os.path.abspath(os.path.join(os.path.abspath(s), os.path.pardir))
    objects_path = os.path.join(sweep_path, "consolidated_objects")
    if not os.path.exists(objects_path):
        return []
    for dname, sdname, files in os.walk(objects_path):
        for dirname in sdname:
            if fnmatch.fnmatch(dirname, "object*"):
                dir_list.append(os.path.join(dname, dirname))
    dir_list = sorted(dir_list, key=sortkey_natural)

    return dir_list

def summarize_objects(data_path):

    sweeps = get_sweep_xmls(data_path)

    images = []
    clouds = []
    poses = []
    detection_type = []
    timestamps = []
    central_images = []
    going_backward = []

    for i, s in enumerate(sweeps):
        print s
        objects = get_objects(s)
        for o in objects:
            print i, o
            object_file = os.path.join(o, "segmented_object.json")
            if not os.path.exists(object_file):
                print object_file, "does not exist, skipping..."
                continue
            with open(object_file) as data_file:
                object_dict = json.load(data_file)['object']
            clouds.append(str(os.path.join(o, "cloud.pcd")))
            images.append([os.path.join(o, im) for im in object_dict['rgb_paths']])
            poses.append(object_dict['pos']['value2'])
            detection_type.append(object_dict['object_type'])
            timestamps.append(i)
            going_backward.append(object_dict['going_backward'])

    for ims in images:
        central_images.append(ims[len(ims)/2])

    poses = np.array(poses)
    timestamps = np.array(timestamps)
    going_backward = np.array(going_backward, dtype=bool)

    print "Images: ", len(images)
    print "Clouds: ", len(clouds)
    print "Poses: ", poses.shape
    print "Types: ", len(detection_type)
    print "Timestamps: ", timestamps.shape
    print "Central images", len(central_images)

    summary_path = os.path.abspath(os.path.join(data_path, "data_summary.npz"))
    np.savez(summary_path, images=images, central_images=central_images, clouds=clouds, poses=poses,
                           detection_type=detection_type, timestamps=timestamps, going_backward=going_backward)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data"
    else:
        summarize_objects(sys.argv[1])
