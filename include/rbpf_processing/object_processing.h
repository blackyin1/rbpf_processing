#ifndef OBJECT_PROCESSING_H
#define OBJECT_PROCESSING_H

#include <rbpf_processing/data_convenience.h>

void consolidate_objects(ObjectVec& objects, FrameVec& frames,
                         const Eigen::Matrix4d& map_pose, float hull_grow = 0.23,
                         float fraction_smaller = 0.4, float fraction_larger = 0.1);
Eigen::Matrix<double, 3, Eigen::Dynamic> compute_object_cloud(SegmentedObject& obj, FrameVec& frames,
                                                              const Eigen::Matrix4d& map_pose);

#endif
