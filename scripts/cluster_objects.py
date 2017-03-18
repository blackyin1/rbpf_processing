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

def cluster_objects(data_path, visualize):

    sweeps = get_sweep_xmls(data_path)

    for s in sweeps:
        print s
        if visualize:
            subprocess.call(['rosrun', 'rbpf_processing', 'consolidate_detections', s, '--visualize'])
        else:
            subprocess.call(['rosrun', 'rbpf_processing', 'consolidate_detections', s])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data (--visualize)"
    elif len(sys.argv) == 2:
        cluster_objects(sys.argv[1], False)
    elif sys.argv[2] == "--visualize":
        cluster_objects(sys.argv[1], True)
    else:
        print "Usage: ", sys.argv[0], " path/to/data (--visualize)"
