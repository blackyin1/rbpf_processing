#!/usr/bin/python

import os
import fnmatch
import sys
import re
import subprocess
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


def propagate_backwards(data_path):

    sweeps = get_sweep_xmls(data_path)
    sweeps.reverse()

    for s in sweeps:
        print s
        subprocess.call(['rosrun', 'rbpf_processing', 'propagate_change_detection', s, '--backwards'])

def propagate_forwards(data_path):

    sweeps = get_sweep_xmls(data_path)

    for s in sweeps:
        print s
        subprocess.call(['rosrun', 'rbpf_processing', 'propagate_change_detection', s])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: ", sys.argv[0], " path/to/data (--backwards)"
    elif len(sys.argv) == 2:
        propagate_forwards(sys.argv[1])
    elif sys.argv[2] == "--backwards":
        propagate_backwards(sys.argv[1])
    else:
        print "Usage: ", sys.argv[0], " path/to/data (--backwards)"
