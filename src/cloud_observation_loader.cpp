#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>

using namespace std;
using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;

class CloudObservationLoader {

public:

    ros::NodeHandle n;
    ros::Subscriber sub;
    ros::Publisher pub;

    CloudObservationLoader() : n()
    {
        pub = n.advertise<sensor_msgs::PointCloud2>("measurement_clouds", 10);
        sub = n.subscribe("cloud_paths", 10, &CloudObservationLoader::callback, this);
    }

    void callback(const std_msgs::String::ConstPtr& str)
    {
        string line(str->data);
        vector<string> paths;
        boost::split(paths, line, boost::is_any_of(","));

        CloudT complete_cloud;
        for (const string& path : paths) {
            CloudT cloud;
            pcl::io::loadPCDFile(path, cloud);
            complete_cloud += cloud;
        }

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(complete_cloud, cloud_msg);
        cloud_msg.header.frame_id = "/map";
        cloud_msg.header.stamp = ros::Time::now();
        pub.publish(cloud_msg);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cloud_observation_loader");

    CloudObservationLoader cl;

    ros::spin();

    return 0;
}
