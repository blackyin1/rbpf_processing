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
    Eigen::Vector3d pos;

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
                cereal::make_nvp("pos", pos),
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
        archive(object_type, going_backward, pos, frames, relative_poses, object_folder, mask_paths, depth_paths, rgb_paths);

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

void add_pos_to_object(SegmentedObject& obj, FrameVec& frames, const Eigen::Matrix4d& map_pose)
{
    double scaling = 1000.0;

    double max_pix = 0;
    size_t max_ind = 0;
    for (size_t i = 0; i < obj.frames.size(); ++i) {
        double pix = cv::sum(obj.masks[i])[0]/255.0;
        if (pix > max_pix) {
            max_pix = pix;
            max_ind = i;
        }
    }

    cv::Mat points;
    cv::findNonZero(obj.masks[max_ind], points);
    cv::Mat depth = frames[obj.frames[max_ind]].depth;
    cv::Mat mask;
    cv::bitwise_and(obj.masks[max_ind], depth > 0, mask);

    double mean_depth = 1.0/scaling*cv::mean(depth, mask)[0];

    Eigen::Vector4d mean_vec;
    mean_vec(3) = 1.0;
    mean_vec(2) = 1.0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (size_t i = 0; i < points.total(); ++i) {
        cv::Point p = points.at<cv::Point>(i);
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_vec(0) = mean_x / double(points.total());
    mean_vec(1) = mean_y / double(points.total());

    mean_vec.head<3>() = frames[obj.frames[max_ind]].K.inverse()*mean_vec.head<3>();
    mean_vec.head<3>() *= mean_depth/mean_vec(2);

    mean_vec = map_pose*frames[obj.frames[max_ind]].pose*mean_vec;

    obj.pos = 1.0/mean_vec(3)*mean_vec.head<3>();
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

void save_object_cloud(SegmentedObject& obj, FrameVec& frames,
                       const Eigen::Matrix4d& map_pose,
                       const string& object_path)
{
    double scaling = 1000.0;

    double max_pix = 0;
    size_t max_ind = 0;
    for (size_t i = 0; i < obj.frames.size(); ++i) {
        double pix = cv::sum(obj.masks[i])[0]/255.0;
        if (pix > max_pix) {
            max_pix = pix;
            max_ind = i;
        }
    }

    cv::Mat locations;   // output, locations of non-zero pixels
    cv::findNonZero(obj.masks[max_ind], locations);
    Eigen::Matrix<double, 4, Eigen::Dynamic> Dp(4, locations.total());
    Eigen::Matrix<int, 3, Eigen::Dynamic> Pp(3, locations.total());
    Dp.row(2).setOnes();

    cout << "Found nonzero with size " << locations.rows << "x" << locations.cols << ", type: " << locations.type() << endl;

    Eigen::Matrix3d Kinv = frames[obj.frames[max_ind]].K.inverse();
    for (size_t j = 0; j < locations.total(); ++j) {
        cv::Point p = locations.at<cv::Point>(j);
        cv::Vec3b c = frames[obj.frames[max_ind]].rgb.at<cv::Vec3b>(p.y, p.x);
        Pp.block<3, 1>(0, j) << c[2], c[1], c[0];
        Dp.block<2, 1>(0, j) << p.x, p.y;
        Dp(3, j) = double(frames[obj.frames[max_ind]].depth.at<uint16_t>(p.y, p.x))/scaling;
    }

    Dp.topRows<3>() = Kinv*Dp.topRows<3>();
    Dp.topRows<3>() = Dp.topRows<3>().array().rowwise() / (Dp.row(2).array() / Dp.row(3).array());
    Dp.row(3).setOnes();
    Dp = map_pose*frames[obj.frames[max_ind]].pose*Dp;
    Dp.topRows<3>() = Dp.topRows<3>().array().rowwise() / Dp.row(3).array();

    CloudT cloud;
    for (size_t j = 0; j < locations.total(); ++j) {
        PointT p; p.getVector3fMap() = Dp.block<3, 1>(0, j).cast<float>();
        p.r = Pp(0, j); p.g = Pp(1, j); p.b = Pp(2, j);
        cloud.points.push_back(p);
    }

    pcl::io::savePCDFileBinary((PathT(object_path) / "cloud.pcd").string(), cloud);
}

void save_objects(ObjectVec& objects, FrameVec& frames, const Eigen::Matrix4d& map_pose,
                  const string& sweep_xml, bool backwards)
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
        cout << "Adding pos to object..." << endl;
        add_pos_to_object(obj, frames, map_pose);
        cout << "Writing object cloud..." << endl;
        save_object_cloud(obj, frames, map_pose, object_path.string());
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
