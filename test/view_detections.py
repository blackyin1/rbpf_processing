#!/usr/bin/python

import os
import fnmatch
import sys
import re
import shutil
import subprocess

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

def view_detections(s):

    sweep_path = os.path.abspath(os.path.join(os.path.abspath(s), os.path.pardir))
    detections = os.path.join(sweep_path, "back_dynamic_clusters.pcd")
    forward = os.path.join(sweep_path, "propagated_dynamic_clusters.pcd")
    backward = os.path.join(sweep_path, "back_propagated_dynamic_clusters.pcd")

    if not os.path.exists(forward):
        forward = ""

    if not os.path.exists(backward):
        backward = ""

    subprocess.call(['pcl_viewer', detections, forward, backward])

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Usage: ", sys.argv[0], " [--files path/to/data, --file /path/to/room.xml]"
    elif sys.argv[1] == "--files":
        sweeps = get_sweep_xmls(sys.argv[2])
        for sweep in sweeps:
            view_detections(sweep)
    elif sys.argv[1] == "--file":
        view_detections(sys.argv[2])
