#include "surfel_type.h"
#include <rbpf_processing/data_convenience.h>
#include <rbpf_processing/xml_convenience.h>

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using SurfelT = SurfelType;
using SurfelCloudT = pcl::PointCloud<SurfelT>;
using PathT = boost::filesystem::path;

void save_segmentation_surfels(const string& sweep_xml)
{
    PathT sweep_path = PathT(sweep_xml).parent_path();
    PathT surfel_path = sweep_path / "surfel_map.pcd";
    PathT segmented_path = sweep_path / "segmented_surfel_map.pcd";

    SurfelCloudT::Ptr surfel_cloud(new SurfelCloudT);
    pcl::io::loadPCDFile(surfel_path.string(), *surfel_cloud);

    Eigen::Matrix<double, 4, Eigen::Dynamic> P(4, surfel_cloud->size());
    for (size_t i = 0; i < surfel_cloud->size(); ++i) {
        P.block<3, 1>(0, i) = surfel_cloud->points[i].getVector3fMap().cast<double>();
    }
    P.row(3).setOnes();

    ObjectVec objects = load_propagated_objects(sweep_xml);

    ObjectVec dummy_objects;
    FrameVec frames;
    Eigen::Matrix4d map_pose;
    // note that these objects should also, eventually include the ones that have been propagated backwards
    tie(dummy_objects, frames, map_pose) = load_objects(sweep_xml, false, false);


    vector<cv::Mat> frame_point_indices;
    vector<cv::Mat> frame_point_masks;
    for (size_t i = 0; i < frames.size(); ++i) {
        // project the surfel map in each of the frames, get mapping point index -> pixels

        Eigen::Matrix4d transform = frames[0].pose.inverse()*frames[i].pose;

        Eigen::Matrix<double, 4, Eigen::Dynamic> Pp = transform.inverse()*P;
        Pp = Pp.array().rowwise() / Pp.row(3).array();
        Pp.topRows<3>() = frames[0].K*Pp.topRows<3>();
        Pp.topRows<2>() = Pp.topRows<2>().array().rowwise() / Pp.row(2).array();

        Eigen::Matrix<int, 2, Eigen::Dynamic> pp = Pp.topRows<2>().cast<int>();

        cv::Mat indices = cv::Mat::zeros(480, 640, CV_32SC1);
        for (size_t j = 0; j < surfel_cloud->size(); ++j) {
            if (pp(0, j) >= 0 && pp(0, j) < 640 && pp(1, j) >= 0 && pp(1, j) < 480) {
                indices.at<int>(pp(1, j), pp(0, j)) = j + 1;
            }
        }

        cv::Mat mask = indices > 0;

        frame_point_indices.push_back(indices);
        frame_point_masks.push_back(mask);
    }

    Eigen::VectorXi types(surfel_cloud->size());
    types.setZero();

    for (SegmentedObject& obj : objects) {

        for (size_t i = 0; i < obj.frames.size(); ++i) {
            cv::Mat mask;
            size_t frame_ind = obj.frames[i];
            cv::bitwise_and(obj.masks[i], frame_point_masks[frame_ind], mask);
            cv::Mat locations;   // output, locations of non-zero pixels
            cv::findNonZero(mask, locations);
            for (size_t j = 0; j < locations.total(); ++j) {
                cv::Point p = locations.at<cv::Point>(j);
                types[frame_point_indices[frame_ind].at<int>(p.y, p.x)-1] = 1 + int(obj.going_backward) + 2*int(obj.object_type == "propagated");
            }
        }

    }

    for (size_t j = 0; j < surfel_cloud->size(); ++j) {
        int obj_type = types[j];
        int r = 0; int g = 0; int b = 0;
        if (obj_type == 0) {
            // do nothing
            continue;
        }
        else if (obj_type == 1) { // detected, forward
            g = 255;
        }
        else if (obj_type == 2) { // detected, backward
            r = 255;
        }
        else if (obj_type == 3) { // propagated, forward
            r = 255; g = 255;
        }
        else if (obj_type == 4) { // propagated, backward
            r = 255; g = 255;
        }
        SurfelT& q = surfel_cloud->at(j);
        uint8_t* rgb = (uint8_t*)(&q.rgba);
        rgb[2] = r;
        rgb[1] = g;
        rgb[0] = b;
    }

    pcl::io::savePCDFileBinary(segmented_path.string(), *surfel_cloud);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " /path/to/room.xml" << endl;
        return 0;
    }

    string sweep_xml(argv[1]);
    save_segmentation_surfels(sweep_xml);

    return 0;
}
