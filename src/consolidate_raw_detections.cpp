#include <rbpf_processing/data_convenience.h>
#include <rbpf_processing/object_processing.h>
#include <rbpf_processing/xml_convenience.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <fstream>

using namespace std;

void visualize_object_clouds(vector<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic> > >& clouds_before,
                             vector<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic> > >& clouds_after)
{
    const int colormap[][3] = {
        {166,206,227},
        {31,120,180},
        {178,223,138},
        {51,160,44},
        {251,154,153},
        {227,26,28},
        {253,191,111},
        {255,127,0},
        {202,178,214},
        {106,61,154},
        {255,255,153},
        {177,89,40},
        {141,211,199},
        {255,255,179},
        {190,186,218},
        {251,128,114},
        {128,177,211},
        {253,180,98},
        {179,222,105},
        {252,205,229},
        {217,217,217},
        {188,128,189},
        {204,235,197},
        {255,237,111},
        {255, 179, 0},
        {128, 62, 117},
        {255, 104, 0},
        {166, 189, 215},
        {193, 0, 32},
        {206, 162, 98},
        {0, 125, 52},
        {246, 118, 142},
        {0, 83, 138},
        {255, 122, 92},
        {83, 55, 122},
        {255, 142, 0},
        {179, 40, 81},
        {244, 200, 0},
        {127, 24, 13},
        {147, 170, 0},
        {89, 51, 21},
        {241, 58, 19},
        {35, 44, 22}
    };

    CloudT::Ptr cloud_before(new CloudT);
    size_t index = 0;
    for (Eigen::Matrix<double, 3, Eigen::Dynamic> matrix_cloud : clouds_before) {
        cloud_before->reserve(cloud_before->size() + matrix_cloud.cols());
        for (size_t i = 0; i < matrix_cloud.cols(); ++i) {
            PointT p;
            p.getVector3fMap() = matrix_cloud.col(i).cast<float>();
            p.r = colormap[index % 44][0];
            p.g = colormap[index % 44][1];
            p.b = colormap[index % 44][2];
            cloud_before->push_back(p);
        }
        ++index;
    }

    CloudT::Ptr cloud_after(new CloudT);
    index = 0;
    for (Eigen::Matrix<double, 3, Eigen::Dynamic> matrix_cloud : clouds_after) {
        cloud_after->reserve(cloud_after->size() + matrix_cloud.cols());
        for (size_t i = 0; i < matrix_cloud.cols(); ++i) {
            PointT p;
            p.getVector3fMap() = matrix_cloud.col(i).cast<float>();
            p.r = colormap[index % 44][0];
            p.g = colormap[index % 44][1];
            p.b = colormap[index % 44][2];
            cloud_after->push_back(p);
        }
        ++index;
    }

    cout << "Points before: " << cloud_before->size() << ", points after: " << cloud_after->size() << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->initCameraParameters();

    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(1, 1, 1, v1);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(cloud_before);
    viewer->addPointCloud<PointT> (cloud_before, rgb1, "sample cloud1", v1);

    int v2(0);
    viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(1, 1, 1, v2);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(cloud_after);
    viewer->addPointCloud<PointT>(cloud_after, rgb2, "sample cloud2", v2);

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud2");

    //viewer->addCoordinateSystem(1.0);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

void consolidate_raw_detections(const string& sweep_xml, bool visualize)
{
    // load all objects and transforms in this sweep that are saved in my format

    ObjectVec objects;
    FrameVec frames;
    Eigen::Matrix4d map_pose;
    // note that these objects should also, eventually include the ones that have been propagated forwards
    tie(objects, frames, map_pose) = load_objects(sweep_xml, false, false, true);

    vector<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic> > > clouds_before;
    if (visualize) {
        // look at how it looked before merging
        for (SegmentedObject obj : objects) {
            clouds_before.push_back(compute_object_cloud(obj, frames, map_pose));
        }
    }

    vector<vector<size_t> > consolidated_indices = consolidate_objects(objects, frames, map_pose);

    cout << "Merged indices:" << endl;
    for (vector<size_t>& v : consolidated_indices) {
        for (size_t i : v) {
            cout << i << " ";
        }
        cout << endl;
    }

    vector<Eigen::Matrix<double, 3, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic> > > clouds_after;
    if (visualize) {
        for (SegmentedObject obj : objects) {
            clouds_after.push_back(compute_object_cloud(obj, frames, map_pose));
        }
        // visualize after merging
        visualize_object_clouds(clouds_before, clouds_after);
    }
    else {
        ofstream indfile;
        boost::filesystem::path indexpath = boost::filesystem::path(sweep_xml).parent_path() / "consolidated_indices.txt";
        indfile.open(indexpath.string());
        for (vector<size_t>& v : consolidated_indices) {
            for (size_t i : v) {
                indfile << i << " ";
            }
            indfile << '\n';
        }
        indfile.close();
    }
}

int main(int argc, char** argv)
{
    bool visualize = false;
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " /path/to/room.xml (--visualize)" << endl;
        return 0;
    }
    else if (argc == 3) {
        if (string(argv[2]) == "--visualize") {
            visualize = true;
        }
    }

    string sweep_xml(argv[1]);
    consolidate_raw_detections(sweep_xml, visualize);

    return 0;
}
