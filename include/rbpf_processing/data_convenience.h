#ifndef DATA_CONVENIENCE_H
#define DATA_CONVENIENCE_H

#include <eigen3/Eigen/Dense>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <tf_conversions/tf_eigen.h>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <eigen_cereal/eigen_cereal.h>

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using SweepT = semantic_map_load_utilties::IntermediateCloudCompleteData<PointT>;

struct SimpleFrame {
    cv::Mat rgb;
    cv::Mat depth;
    double time;
    Eigen::Matrix3d K;
    Eigen::Matrix4d pose;
};

using FrameVec = std::vector<SimpleFrame>;

std::string num_str(size_t i)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << i;
    return std::string(ss.str());
}

struct SegmentedObject {

    std::string object_type; // can be "detected", "propagated" and "occluded"
    bool going_backward; // from backwards propagation or forward propagation
    Eigen::Vector3d pos;

    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> depth_masks; // optional, for projection of object from other frame
    std::vector<cv::Mat> cropped_rgbs; // optional, for training the CNN features
    std::vector<size_t> frames; // index in the FrameVec return
    PoseVec relative_poses; // isn't this a bit unnecessary? Well, if they are registered maybe but... optional
    std::string object_folder;

    double dims[3];

    template <class Archive>
    void save(Archive& archive) const
    {
        boost::filesystem::path object_path(object_folder);

        std::vector<std::string> depth_paths, mask_paths, rgb_paths;
        for (size_t i = 0; i < depth_masks.size(); ++i) {
            std::string depth_path = std::string("depth_mask") + num_str(i) + ".png";
            depth_paths.push_back(depth_path);
            cv::imwrite((object_path / depth_path).string(), depth_masks[i]);
        }
        for (size_t i = 0; i < masks.size(); ++i) {
            std::string mask_path = std::string("mask") + num_str(i) + ".png";
            mask_paths.push_back(mask_path);
            cv::imwrite((object_path / mask_path).string(), masks[i]);
        }
        for (size_t i = 0; i < cropped_rgbs.size(); ++i) {
            std::string rgb_path = std::string("cropped_rgb") + num_str(i) + ".jpeg";
            rgb_paths.push_back(rgb_path);
            cv::imwrite((object_path / rgb_path).string(), cropped_rgbs[i]);
        }

        archive(cereal::make_nvp("object_type", object_type),
                cereal::make_nvp("going_backward", going_backward),
                cereal::make_nvp("pos", pos),
                cereal::make_nvp("frames", frames),
                cereal::make_nvp("relative_poses", relative_poses),
                cereal::make_nvp("segment_folder", object_folder),
                cereal::make_nvp("mask_paths", mask_paths),
                cereal::make_nvp("depth_paths", depth_paths),
                cereal::make_nvp("rgb_paths", rgb_paths),
                cereal::make_nvp("dims", dims));

    }

    template <class Archive>
    void load(Archive& archive)
    {
        std::vector<std::string> depth_paths, mask_paths, rgb_paths;
        archive(object_type, going_backward, pos, frames, relative_poses, object_folder, mask_paths, depth_paths, rgb_paths, dims);

        boost::filesystem::path object_path(object_folder);

        for (size_t i = 0; i < depth_paths.size(); ++i) {
            depth_masks.push_back(cv::imread((object_path / depth_paths[i]).string(), CV_LOAD_IMAGE_GRAYSCALE));
        }
        for (size_t i = 0; i < mask_paths.size(); ++i) {
            masks.push_back(cv::imread((object_path / mask_paths[i]).string(), CV_LOAD_IMAGE_GRAYSCALE));
        }
        for (size_t i = 0; i < rgb_paths.size(); ++i) {
            cropped_rgbs.push_back(cv::imread((object_path / rgb_paths[i]).string()));
        }
    }
};

using ObjectVec = std::vector<SegmentedObject>;

void add_cropped_rgb_to_object(SegmentedObject& obj, FrameVec& frames);
void add_pos_to_object(SegmentedObject& obj, FrameVec& frames, const Eigen::Matrix4d& map_pose);
PoseVec load_transforms_for_data(SweepT& data);
CloudT::Ptr save_object_cloud(SegmentedObject& obj, FrameVec& frames,
                              const Eigen::Matrix4d& map_pose,
                              const std::string& object_path);
void save_complete_propagated_cloud(std::vector<CloudT::Ptr>& clouds, const std::string& sweep_xml, bool backwards);
void save_objects(ObjectVec& objects, FrameVec& frames, const Eigen::Matrix4d& map_pose,
                  const std::string& sweep_xml, bool backwards, bool save_cloud = true);
void save_complete_objects(ObjectVec& objects, FrameVec& frames,
                           const Eigen::Matrix4d& map_pose, const std::string& sweep_xml);
ObjectVec load_propagated_objects(const std::string& sweep_xml, bool do_filter = false, bool backwards = false);

#endif // DATA_CONVENIENCE_H
