#include <eigen3/Eigen/Dense>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <tf_conversions/tf_eigen.h>
#include <rbpf_processing/data_convenience.h>

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using SweepT = semantic_map_load_utilties::IntermediateCloudCompleteData<PointT>;

void visualize_sweep_registration(const string& sweep_xml, bool backwards)
{
    PoseVec current_transforms;
    string previous_xml;
    tie(previous_xml, current_transforms) = read_previous_sweep_params(sweep_xml, backwards);

    SweepT previous_data = semantic_map_load_utilties::loadIntermediateCloudsCompleteDataFromSingleSweep<PointT>(previous_xml);

    PoseVec previous_transforms = load_transforms_for_data(previous_data);

    SweepT current_data = semantic_map_load_utilties::loadIntermediateCloudsCompleteDataFromSingleSweep<PointT>(sweep_xml);

    CloudT::Ptr combined_cloud(new CloudT);

    for (size_t i = 0; i < previous_transforms.size(); ++i) {
        CloudT::Ptr transformed_cloud(new CloudT);
        pcl::transformPointCloud(*previous_data.vIntermediateRoomClouds[i], *transformed_cloud, previous_transforms[i]);
        *combined_cloud += *transformed_cloud;
    }

    for (size_t i = 0; i < current_transforms.size(); ++i) {
        CloudT::Ptr transformed_cloud(new CloudT);
        pcl::transformPointCloud(*current_data.vIntermediateRoomClouds[i], *transformed_cloud, current_transforms[i]);
        *combined_cloud += *transformed_cloud;
    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(1, 1, 1);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(combined_cloud);
    viewer->addPointCloud<PointT>(combined_cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    //viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

int main(int argc, char** argv)
{
    bool backwards = false;
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " /path/to/room.xml (--backwards)" << endl;
        return 0;
    }
    else if (argc == 3) {
        if (string(argv[2]) == "--backwards") {
            backwards = true;
        }
    }

    string sweep_xml(argv[1]);
    visualize_sweep_registration(sweep_xml, backwards);

    return 0;
}
