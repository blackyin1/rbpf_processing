#ifndef XML_CONVENIENCE_H
#define XML_CONVENIENCE_H

#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <rbpf_processing/data_convenience.h>

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using RoomT = SimpleXMLParser<PointT>::RoomData;
using FrameVec = std::vector<SimpleFrame>;

std::pair<std::string, PoseVec> read_previous_sweep_params(const std::string& sweep_xml, bool backwards);
Eigen::Matrix4d getPose(QXmlStreamReader& xmlReader);
std::pair<FrameVec, PoseVec> readViewXML(const std::string& roomLogName, const std::string& xmlFile);
std::pair<FrameVec, PoseVec> load_frames_poses(RoomT& data);
std::tuple<ObjectVec, FrameVec, Eigen::Matrix4d> load_objects(const std::string& path, bool backwards = false, bool load_propagated = false, bool load_detected = true);
std::pair<FrameVec, Eigen::Matrix4d> load_frames_pose(const std::string& path);

#endif // XML_CONVENIENCE_H
