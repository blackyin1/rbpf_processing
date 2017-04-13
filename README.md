# rbpf_processing
For data collection. detecting objects through change detection and extracting CNN features for tracking in the rbpf_mtt framework

## Waypoint to visit during the daily routine:

Four times a day, visit the following nodes:

* WayPoint33 - hallway area
* WayPoint34 - master student room at entrance
* WayPoint29 - kitchen
* WayPoint27 - in front of Kitchen door
* WayPoint1 - in front of charging station
* WayPoint9 - meeting room
* WayPoint22 - in front of Jana's office
* WayPoint6 - Nils' room
* WayPoint7 - Rares' room
* WayPoint17 - a bit down the hallway
* WayPoint16 - the end of the hallway
* WayPoint4 - large office at end of hallway

## Commands to run to process the data
```
rosrun quasimodo_brain metaroom_additional_view_processing -once -notSendPrev -resegment -files /path/to/data
```
```
rosrun quasimodo_brain metaroom_additional_view_processing -once -notSendPrev -backwards -resegment -files /path/to/data
```
```
rosrun rbpf_processing rbpf_propagation.py /path/to/data
```
```
rosrun rbpf_processing rbpf_propagation.py /path/to/data --backwards
```
```
rosrun rbpf_processing cluster_objects.py /path/to/data
```
```
rosrun rbpf_processing summarize_objects.py /path/to/data
```
```
rosrun rbpf_processing extract_features.py /path/to/data
```
```
rosrun rbpf_processing reduce_features.py /path/to/data
```
```
rosrun rbpf_processing rbpf_conversion.py /path/to/data
```
