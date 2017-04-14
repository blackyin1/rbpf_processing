#!/usr/bin/python

import os
import fnmatch
import sys
import re
import subprocess
import json
import numpy as np
import xml.etree.ElementTree as ET
from os.path import relpath
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

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def summarize_objects(data_path):

    sweeps = get_sweep_xmls(data_path)

    images = []
    clouds = []
    poses = []
    detection_type = []
    timestamps = []
    central_images = []
    going_backward = []
    waypoint_names = []
    location_ids = []
    dims = []

    for i, s in enumerate(sweeps):
        tree = ET.parse(s)
        root = tree.getroot()
        room_id_element = root.find("RoomStringId")
        location_id = get_trailing_number(room_id_element.text)
        print room_id_element.text
        print location_id
        print s
        objects = get_objects(s)
        for o in objects:
            rel_path = relpath(o, data_path)
            #print rel_path
            print i, o
            object_file = os.path.join(o, "segmented_object.json")
            if not os.path.exists(object_file):
                print object_file, "does not exist, skipping..."
                continue
            with open(object_file) as data_file:
                object_dict = json.load(data_file)['object']
            clouds.append(str(os.path.join(rel_path, "cloud.pcd")))
            images.append([os.path.join(rel_path, im) for im in object_dict['rgb_paths']])
            poses.append(object_dict['pos']['value2'])
            detection_type.append(object_dict['object_type'])
            timestamps.append(i)
            going_backward.append(object_dict['going_backward'])
            waypoint_names.append(room_id_element.text)
            location_ids.append(location_id)
            dims_dict = object_dict['dims']
            dims.append([dims_dict['value0'], dims_dict['value1'], dims_dict['value2']])

    for ims in images:
        central_images.append(ims[len(ims)/2])

    poses = np.array(poses)
    timestamps = np.array(timestamps)
    going_backward = np.array(going_backward, dtype=bool)
    location_ids = np.array(location_ids, dtype=int)
    dims = np.array(dims)

    print "Images: ", len(images)
    print "Clouds: ", len(clouds)
    print "Poses: ", poses.shape
    print "Types: ", len(detection_type)
    print "Timestamps: ", timestamps.shape
    print "Central images", len(central_images)

    summary_path = os.path.abspath(os.path.join(data_path, "data_summary.npz"))
    np.savez(summary_path, images=images, central_images=central_images, clouds=clouds, poses=poses, detection_type=detection_type,
                           timestamps=timestamps, going_backward=going_backward, location_ids=location_ids, dims=dims)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data"
    else:
        summarize_objects(sys.argv[1])
