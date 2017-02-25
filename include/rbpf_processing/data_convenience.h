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

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using SweepT = semantic_map_load_utilties::IntermediateCloudCompleteData<PointT>;

struct SimpleFrame {
    cv::Mat rgb;
    cv::Mat depth;
    double time;
    Eigen::Matrix3d K;
    Eigen::Matrix4d pose;
};

using FrameVec = vector<SimpleFrame>;

string num_str(size_t i)
{
    stringstream ss;
    ss << setfill('0') << setw(4) << i;
    return string(ss.str());
}

struct SegmentedObject {

    string object_type; // can be "detected", "propagated" and "occluded"
    bool going_backward; // from backwards propagation or forward propagation

    vector<cv::Mat> masks;
    vector<cv::Mat> depth_masks; // optional, for projection of object from other frame
    vector<cv::Mat> cropped_rgbs; // optional, for training the CNN features
    vector<size_t> frames; // index in the FrameVec return
    PoseVec relative_poses; // isn't this a bit unnecessary? Well, if they are registered maybe but... optional
    string object_folder;

    template <class Archive>
    void save(Archive& archive) const
    {
        boost::filesystem::path object_path(object_folder);

        vector<string> depth_paths, mask_paths, rgb_paths;
        for (size_t i = 0; i < depth_masks.size(); ++i) {
            string depth_path = string("depth_mask") + num_str(i) + ".png";
            depth_paths.push_back(depth_path);
            cv::imwrite((object_path / depth_path).string(), depth_masks[i]);
        }
        for (size_t i = 0; i < masks.size(); ++i) {
            string mask_path = string("mask") + num_str(i) + ".png";
            mask_paths.push_back(mask_path);
            cv::imwrite((object_path / mask_path).string(), masks[i]);
        }
        for (size_t i = 0; i < cropped_rgbs.size(); ++i) {
            string rgb_path = string("cropped_rgb") + num_str(i) + ".jpeg";
            rgb_paths.push_back(rgb_path);
            cv::imwrite((object_path / rgb_path).string(), cropped_rgbs[i]);
        }

        archive(cereal::make_nvp("object_type", object_type),
                cereal::make_nvp("going_backward", going_backward),
                cereal::make_nvp("frames", frames),
                cereal::make_nvp("relative_poses", relative_poses),
                cereal::make_nvp("segment_folder", object_folder),
                cereal::make_nvp("mask_paths", mask_paths),
                cereal::make_nvp("depth_paths", depth_paths),
                cereal::make_nvp("rgb_paths", rgb_paths));

    }

    template <class Archive>
    void load(Archive& archive)
    {
        boost::filesystem::path object_path(object_folder);

        vector<string> depth_paths, mask_paths, rgb_paths;
        archive(object_type, going_backward, frames, relative_poses, object_folder, mask_paths, depth_paths, rgb_paths);

        for (size_t i = 0; i < depth_paths.size(); ++i) {
            depth_masks.push_back(cv::imread((object_path / depth_paths[i]).string()));
        }
        for (size_t i = 0; i < mask_paths.size(); ++i) {
            masks.push_back(cv::imread((object_path / mask_paths[i]).string()));
        }
        for (size_t i = 0; i < rgb_paths.size(); ++i) {
            cropped_rgbs.push_back(cv::imread((object_path / rgb_paths[i]).string()));
        }
    }
};

using ObjectVec = vector<SegmentedObject>;

void add_cropped_rgb_to_object(SegmentedObject& obj, FrameVec& frames)
{
    for (size_t i = 0; i < obj.frames.size(); ++i) {
        cv::Mat points;
        cv::findNonZero(obj.masks[i], points);
        cv::Rect min_rect = cv::boundingRect(points);
        cv::Mat cropped = frames[obj.frames[i]].rgb(min_rect);
        obj.cropped_rgbs.push_back(cropped);
    }
}

PoseVec load_transforms_for_data(SweepT& data)
{
    PoseVec transforms;
    for (tf::StampedTransform t : data.vIntermediateRoomCloudTransformsRegistered) {
        Eigen::Affine3d e;
        tf::transformTFToEigen(t, e);
        transforms.push_back(e.matrix());
    }
    return transforms;
}

void save_objects(ObjectVec& objects, FrameVec& frames, const string& sweep_xml, bool backwards)
{
    if (objects.empty()) {
        return;
    }

    cout << "Saving objects, creating directory..." << endl;

    PathT objects_path = PathT(sweep_xml).parent_path() / "consolidated_objects";
    if (!boost::filesystem::exists(objects_path)) {
        boost::filesystem::create_directory(objects_path);
    }

    cout << "Creating object subdirectories..." << endl;

    // how many objects are already saved in this folder?
    size_t i = 0;
    while (true) {
        PathT object_path = objects_path / (string("object") + num_str(i));
        if (!boost::filesystem::exists(object_path)) {
            break;
        }
        ++i;
    }

    // ok, save this in a format that we can use to extract the CNN features (JPEG FTW)
    // one thing to note: we'll have to do another pass where we get all the image paths
    // fortunately, there's python
    for (SegmentedObject& obj : objects) {
        cout << "Saving object " << i << endl;
        // let's create a folder for every object
        PathT object_path = objects_path / (string("object") + num_str(i));
        boost::filesystem::create_directory(object_path);
        obj.object_folder = object_path.string();
        PathT object_file = object_path / "segmented_object.json";
        cout << "Adding rgb images..." << endl;
        add_cropped_rgb_to_object(obj, frames);
        cout << "Writing object..." << endl;
        ofstream out(object_file.string());
        {
            cereal::JSONOutputArchive archive_o(out);
            archive_o(cereal::make_nvp("object", obj));
        }
        ++i;
    }

    cout << "Done saving objects..." << endl;
}

#endif // DATA_CONVENIENCE_H
