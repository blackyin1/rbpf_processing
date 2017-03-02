#include <eigen3/Eigen/Dense>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <eigen_cereal/eigen_cereal.h>
#include <rbpf_processing/data_convenience.h>
#include <rbpf_processing/xml_convenience.h>

using namespace std;

// this is obsolete
void obsolete_save_objects(const string& room_path, ObjectVec& objects, FrameVec& frames)
{
    boost::filesystem::path folder_path = boost::filesystem::path(room_path).parent_path();

    size_t counter = 0;
    for (SegmentedObject& obj : objects) {
        boost::filesystem::path obj_path = folder_path / (string("object_images") + to_string(counter));
        boost::filesystem::create_directory(obj_path);
        for (size_t i = 0; i < obj.frames.size(); ++i) {
            cv::imwrite((obj_path / (string("mask") + to_string(i) + ".png")).string(), obj.masks[i]);
            cv::imwrite((obj_path / (string("rgb") + to_string(i) + ".png")).string(), frames[obj.frames[i]].rgb);
            boost::filesystem::path camera_path = obj_path / (string("rgb") + to_string(i) + ".json");
            boost::filesystem::path pose_path = obj_path / (string("pose") + to_string(i) + ".json");
            boost::filesystem::path rect_path = obj_path / (string("bounding_box") + to_string(i) + ".json");
            ofstream camera_out(camera_path.string());
            {
                cereal::JSONOutputArchive archive_o(camera_out);
                archive_o(frames[obj.frames[i]].K);
            }
            ofstream pose_out(pose_path.string());
            {
                cereal::JSONOutputArchive archive_o(pose_out);
                archive_o(obj.relative_poses[i]);
            }
            cv::Mat points;
            cv::findNonZero(obj.masks[i], points);
            cv::Rect min_rect = cv::boundingRect(points);

            map<string, int> rect_map;
            rect_map["x"] = min_rect.x;
            rect_map["y"] = min_rect.y;
            rect_map["width"] = min_rect.width;
            rect_map["height"] = min_rect.height;
            ofstream rect_out(rect_path.string());
            {
                cereal::JSONOutputArchive archive_o(rect_out);
                archive_o(rect_map);
            }
            // use cereal and eigen_cereal for storing the pose and camera matrix although K is not important
        }
        ++counter;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " /path/to/room.xml" << endl;
        return 0;
    }

    string room_path(argv[1]);

    tuple<ObjectVec, FrameVec, Eigen::Matrix4d> objects = load_objects(room_path);
    obsolete_save_objects(room_path, get<0>(objects), get<1>(objects));

    return 0;
}
