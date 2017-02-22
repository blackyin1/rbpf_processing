#include <eigen3/Eigen/Dense>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <eigen_cereal/eigen_cereal.h>

using namespace std;
using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using LabelT = semantic_map_load_utilties::LabelledData<PointT>;

struct DetectionData {
    int x, y, width, height;
    string label;
    Eigen::Matrix4d pose;
    Eigen::Matrix3d K;

    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(cereal::make_nvp("x", x),
                cereal::make_nvp("y", y),
                cereal::make_nvp("width", width),
                cereal::make_nvp("height", height),
                cereal::make_nvp("label", label),
                cereal::make_nvp("pose", pose),
                cereal::make_nvp("K", K));
    }
};

pair<cv::Mat, cv::Mat> sweep_get_rgbd_at(const boost::filesystem::path& sweep_xml, size_t i)
{
    stringstream ss;
    ss << "intermediate_cloud" << std::setfill('0') << std::setw(4) << i << ".pcd";
    boost::filesystem::path cloud_path = sweep_xml.parent_path() / ss.str();
    CloudT::Ptr cloud(new CloudT);
    pcl::io::loadPCDFile(cloud_path.string(), *cloud);
    pair<cv::Mat, cv::Mat> images = SimpleXMLParser<PointT>::createRGBandDepthFromPC(cloud);
    return images;
}

pair<vector<string>, vector<string> > process_sweep(const string& room_xml)
{
    vector<string> images;
    vector<string> string_labels;
    LabelT labels = semantic_map_load_utilties::loadLabelledDataFromSingleSweep<PointT>(room_xml);

    boost::filesystem::path folder_path = boost::filesystem::path(room_xml).parent_path();
    boost::filesystem::path obj_path = folder_path / "object_images";
    boost::filesystem::create_directory(obj_path);

    size_t N = labels.objectClouds.size();
    for (size_t i = 0; i < N; ++i) {
        // labels.objectClouds, labels.objectLabels, labels.objectImages, labels.objectMasks, labels.objectScanIndices
        cout << "Processing object: " << i << ", with label: " << labels.objectLabels[i] << endl;

        DetectionData data;

        cv::Mat points;
        cv::findNonZero(labels.objectMasks[i], points);
        cv::Rect min_rect = cv::boundingRect(points);
        data.x = min_rect.x;
        data.y = min_rect.y;
        data.width = min_rect.width;
        data.height = min_rect.height;
        data.label = labels.objectLabels[i];
        string_labels.push_back(labels.objectLabels[i]);
        ofstream out((obj_path / (string("detection_data") + to_string(i) + ".json")).string());
        {
            cereal::JSONOutputArchive archive_o(out);
            archive_o(data);
        }

        size_t scan_index = labels.objectScanIndices[i];
        pair<cv::Mat, cv::Mat> rgbd = sweep_get_rgbd_at(room_xml, scan_index);

        //cv::Mat image = labels.objectImages[i](min_rect);
        cv::Mat image = rgbd.first(min_rect);
        boost::filesystem::path image_path = obj_path / (string("rgb_image") + to_string(i) + ".jpeg");
        cv::imwrite(image_path.string(), image);
        images.push_back(image_path.string());
    }

    return make_pair(images, string_labels);
}

void process_sweeps(const string& sweep_folder)
{
    vector<string> images;
    vector<string> labels;

    vector<string> sweep_xmls = semantic_map_load_utilties::getSweepXmls<PointT>(sweep_folder, true);
    for (const string& xml : sweep_xmls) {
        cout << "Running for sweep: " << xml << endl;
        vector<string> sweep_images;
        vector<string> sweep_labels;
        tie(sweep_images, sweep_labels) = process_sweep(xml);
        images.insert(images.end(), sweep_images.begin(), sweep_images.end());
        labels.insert(labels.end(), sweep_labels.begin(), sweep_labels.end());
    }

    ofstream images_out((boost::filesystem::path(sweep_folder) / "feature_rgb_images.json").string());
    {
        cereal::JSONOutputArchive archive_o(images_out);
        archive_o(images);
    }
    ofstream labels_out((boost::filesystem::path(sweep_folder) / "feature_labels.json").string());
    {
        cereal::JSONOutputArchive archive_o(labels_out);
        archive_o(labels);
    }
}

// in this file, we will simply process the metaroom labeled
// images to extract cropped images, save as jpeg in one folder
// for every sweep and also save the poses and object labels
// these will then be fed into the CNN feature extractor

// the goal with implementing this is to visualize a TSDE dimension
// reduction in 2 dimensions with the labels of the objects as colors
// hopefully, this can illustrate how well we manage to separate the different
// types of object instances

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " /path/to/data" << endl;
        return 0;
    }

    process_sweeps(string(argv[1]));

    return 0;
}
