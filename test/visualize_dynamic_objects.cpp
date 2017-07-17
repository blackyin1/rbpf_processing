#include <rbpf_processing/data_convenience.h>
#include <rbpf_processing/object_processing.h>
#include <rbpf_processing/xml_convenience.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

void visualize(CloudT::Ptr& cloud)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(1, 1, 1);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
    viewer->addPointCloud<PointT>(cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    //viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void visualize_dynamic_objects(const string& sweep_xml)
{
    // load all objects and transforms in this sweep that are saved in my format

    ObjectVec objects;
    FrameVec frames;
    Eigen::Matrix4d map_pose;
    // note that these objects should also, eventually include the ones that have been propagated forwards
    tie(objects, frames, map_pose) = load_objects(sweep_xml, false, false, true);

    if (objects.empty()) {
        return;
    }

    CloudT::Ptr cloud(new CloudT);
    for (SegmentedObject obj : objects) {
        CloudT::Ptr object_cloud = get_object_cloud(obj, frames, map_pose);
        cloud->reserve(cloud->size() + object_cloud->size());
        for (PointT p : object_cloud->points) {
            cloud->push_back(p);
        }
    }

    visualize(cloud);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " /path/to/room.xml" << endl;
        return 0;
    }
    string sweep_xml(argv[1]);
    visualize_dynamic_objects(sweep_xml);

    return 0;

}
