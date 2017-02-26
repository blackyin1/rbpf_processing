#!/usr/bin/python

import os
import fnmatch
import sys
import re
import shutil

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

def remove_objects(data_path):

    sweeps = get_sweep_xmls(data_path)

    for s in sweeps:
        sweep_path = os.path.abspath(os.path.join(os.path.abspath(s), os.path.pardir))
        objects_path = os.path.join(sweep_path, "consolidated_objects")
        if os.path.exists(objects_path):
            print "Deleting: ", objects_path
            #os.rmdir(objects_path)
            shutil.rmtree(objects_path)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data (--backwards)"
    else:
        remove_objects(sys.argv[1])
