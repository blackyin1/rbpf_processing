#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <boost/filesystem.hpp>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;
using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using RoomT = SimpleXMLParser<PointT>::RoomData;

static const int colormap[][3] = {
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

class CloudObservationLoader {

public:

    ros::NodeHandle n;
    ros::Subscriber sub;
    ros::Publisher pub;
    ros::Publisher sweep_pub;
    bool load_rooms;
    bool color_clouds;

    CloudObservationLoader() : n()
    {
        ros::NodeHandle pn("~");
        pn.param<bool>("load_rooms", load_rooms, true);
        pn.param<bool>("color_clouds", color_clouds, true);
        pub = n.advertise<sensor_msgs::PointCloud2>("measurement_clouds", 50);
        sweep_pub = n.advertise<sensor_msgs::PointCloud2>("complete_cloud", 50);
        sub = n.subscribe("cloud_paths", 50, &CloudObservationLoader::callback, this);
    }

    void callback(const std_msgs::String::ConstPtr& str)
    {
        string line(str->data);
        if (line.empty()) {
            CloudT cloud;
            PointT p;
            p.getVector3fMap().setZero();
            p.rgba = 0.0;
            cloud.push_back(p);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(cloud, cloud_msg);
            cloud_msg.header.frame_id = "/map";
            cloud_msg.header.stamp = ros::Time::now();
            pub.publish(cloud_msg);
            return;
        }
        vector<string> paths;
        boost::split(paths, line, boost::is_any_of(","));

        PathT sweep_path;

        CloudT complete_cloud;
        size_t counter = 0;
        for (const string& path : paths) {

            PathT current_path = PathT(path).parent_path().parent_path().parent_path();
            if (sweep_path.empty()) {
                sweep_path = current_path;
            }
            else if (sweep_path != current_path) {
                cout << "Got clouds from different sweeps " << sweep_path.string() << " and " << current_path.string() << endl;
                sweep_path = current_path;
            }

            CloudT cloud;
            pcl::io::loadPCDFile(path, cloud);
            if (color_clouds) {
                for (PointT& p : cloud.points) {
                    p.r = colormap[counter % 43][2];
                    p.g = colormap[counter % 43][1];
                    p.b = colormap[counter % 43][0];
                }
            }
            complete_cloud += cloud;
            ++counter;
        }

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(complete_cloud, cloud_msg);
        cloud_msg.header.frame_id = "/map";
        cloud_msg.header.stamp = ros::Time::now();
        pub.publish(cloud_msg);

        if (load_rooms && !sweep_path.empty()) {
            PathT cloud_path = sweep_path / "complete_cloud.pcd";
            if (!boost::filesystem::exists(cloud_path)) {
                return;
            }
            RoomT roomData  = SimpleXMLParser<PointT>::loadRoomFromXML((sweep_path / "room.xml").string(), vector<string>{"RoomIntermediateCloud"}, false, false);
            Eigen::Affine3d e;
            //tf::transformTFToEigen(roomData.vIntermediateRoomCloudTransforms[0], e); // NOTE: this is how it used to be done
            tf::transformTFToEigen(roomData.vIntermediateRoomCloudTransformsRegistered[0], e);

            //cout << "Transform: " << e.matrix() << endl;

            CloudT::Ptr cloud(new CloudT);
            pcl::io::loadPCDFile(cloud_path.string(), *cloud);
            pcl::VoxelGrid<PointT> sor;
            sor.setInputCloud(cloud);
            sor.setLeafSize(0.02f, 0.02f, 0.02f);
            CloudT cloud_filtered;
            sor.filter(cloud_filtered);

            CloudT cloud_transformed;
            pcl::transformPointCloud(cloud_filtered, cloud_transformed, e.matrix());

            sensor_msgs::PointCloud2 sweep_msg;
            pcl::toROSMsg(cloud_filtered, sweep_msg);
            sweep_msg.header.frame_id = "/map";
            sweep_msg.header.stamp = ros::Time::now();
            sweep_pub.publish(sweep_msg);
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cloud_observation_loader");

    CloudObservationLoader cl;

    ros::spin();

    return 0;
}
