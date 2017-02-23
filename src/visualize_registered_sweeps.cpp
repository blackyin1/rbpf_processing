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

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using SweepT = semantic_map_load_utilties::IntermediateCloudCompleteData<PointT>;

pair<string, PoseVec> read_previous_sweep_params(const string& sweep_xml, bool backwards)
{
    //map<string, map<string, string> > loaded_file;
    PathT sweep_folder = PathT(sweep_xml).parent_path();

    PathT filename;
	if (backwards) {
	 	filename = sweep_folder / "back_relative_model_poses.xml";
	}
	else {
		filename = sweep_folder / "relative_model_poses.xml";
	}

    cout << "Reading file: " << filename.string() << endl;

    /*
    std::ifstream in(filename.string());
    {
        cereal::XMLInputArchive archive_i(in);
        archive_i(loaded_file);
    }

    for (const pair<string, map<string, string > >& p : loaded_file) {
        cout << "Loaded map: " << p.first << endl;
        for (const pair<string, string>& q : p.second) {
            cout << "Loaded value: " << q.first << " with value: " << q.second << endl;
        }
    }
    */

    QFile file(filename.c_str());
	if (!file.exists()){
		cout << "Could not open file " << filename.string() << endl;
		exit(-1);
	}
	file.open(QIODevice::ReadOnly);
    QXmlStreamReader xmlReader(&file);

    string previous_sweep_xml;
    vector<string> frame_transform_strings;

    while (!xmlReader.atEnd() && !xmlReader.hasError()) {
        QXmlStreamReader::TokenType token = xmlReader.readNext();
        if (token == QXmlStreamReader::StartDocument) {
            continue;
        }

        if (xmlReader.hasError()) {
            ROS_ERROR("XML error: %s",xmlReader.errorString().toStdString().c_str());
            break;
        }

        if (token == QXmlStreamReader::StartElement) {
            if (xmlReader.name() == "ComparedSweep") {
                QXmlStreamAttributes attributes = xmlReader.attributes();
                if (attributes.hasAttribute("Path")) {
                    previous_sweep_xml = attributes.value("Path").toString().toStdString();
                }
                else {
                    break;
                }
            }
            else if (xmlReader.name() == "Poses") {
                QXmlStreamAttributes attributes = xmlReader.attributes();
                for (size_t i = 0; i < 17; ++i) {
                    QString FrameAttribute = QString("Frame") + QString::number(i);
                    if (attributes.hasAttribute(FrameAttribute)) {
                        frame_transform_strings.push_back(attributes.value(FrameAttribute).toString().toStdString());
                    }
                }
            }
        }
    }

    cout << "Previous sweep xml: " << previous_sweep_xml << endl;
    PoseVec sweep_transforms;
    for (const string& transform_string : frame_transform_strings) {
        Eigen::Matrix4d T;
        string input = transform_string;
        std::replace(input.begin(), input.end(), ',', ' ');
        vector<double> inputs;
        istringstream in(input);
        std::copy(std::istream_iterator<double>(in), std::istream_iterator<double>(), std::back_inserter(inputs));
        size_t i = 0;
        for (size_t y = 0; y < 4; ++y) {
			for (size_t x = 0; x < 4; ++x) {
                T(y, x) = inputs[i];
                ++i;
			}
        }
        sweep_transforms.push_back(T);
        //cout << "Got transform " << T << endl;
    }

    return make_pair(previous_sweep_xml, sweep_transforms);
}

void visualize_sweep_registration(const string& sweep_xml, bool backwards)
{
    PoseVec current_transforms;
    string previous_xml;
    tie(previous_xml, current_transforms) = read_previous_sweep_params(sweep_xml, backwards);

    SweepT previous_data = semantic_map_load_utilties::loadIntermediateCloudsCompleteDataFromSingleSweep<PointT>(previous_xml);

    PoseVec previous_transforms;
    for (tf::StampedTransform t : previous_data.vIntermediateRoomCloudTransformsRegistered) {
        Eigen::Affine3d e;
        tf::transformTFToEigen(t, e);
        previous_transforms.push_back(e.matrix());
    }

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
