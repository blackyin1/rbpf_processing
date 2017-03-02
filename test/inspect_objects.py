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

def inspect_objects(data_path):

    sweeps = get_sweep_xmls(data_path)

    backward_detected = 0
    forward_detected = 0
    backward_propagated = 0
    forward_propagated = 0

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
            if object_dict['going_backward']:
                if object_dict['object_type'] == "detected":
                    backward_detected += 1
                elif object_dict['object_type'] == "propagated":
                    backward_propagated += 1
            else:
                if object_dict['object_type'] == "detected":
                    forward_detected += 1
                elif object_dict['object_type'] == "propagated":
                    forward_propagated += 1

    print "Nbr backward detected: ", backward_detected
    print "Nbr forward detected: ", forward_detected
    print "Nbr backward propagated: ", backward_propagated
    print "Nbr forward propagated: ", forward_propagated


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data"
    else:
        inspect_objects(sys.argv[1])
