#include <eigen3/Eigen/Dense>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <eigen_cereal/eigen_cereal.h>

using namespace std;

struct SimpleFrame {
    cv::Mat rgb;
    cv::Mat depth;
    double time;
    Eigen::Matrix3d K;
    Eigen::Matrix4d pose;
};

using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using FrameVec = vector<SimpleFrame>;

struct SegmentedObject {
    vector<cv::Mat> masks;
    vector<size_t> frames; // index in the FrameVec return
    PoseVec relative_poses;
};

using ObjectVec = vector<SegmentedObject>;

Eigen::Matrix4d getPose(QXmlStreamReader& xmlReader)
{
	QXmlStreamReader::TokenType token = xmlReader.readNext();//Translation
	QString elementName = xmlReader.name().toString();

	token = xmlReader.readNext();//fx
	double tx = atof(xmlReader.readElementText().toStdString().c_str());

	token = xmlReader.readNext();//fy
	double ty = atof(xmlReader.readElementText().toStdString().c_str());

	token = xmlReader.readNext();//cx
	double tz = atof(xmlReader.readElementText().toStdString().c_str());

	token = xmlReader.readNext();//Translation
	elementName = xmlReader.name().toString();

	token = xmlReader.readNext();//Rotation
	elementName = xmlReader.name().toString();

	token = xmlReader.readNext();//qw
	double qw = atof(xmlReader.readElementText().toStdString().c_str());

	token = xmlReader.readNext();//qx
	double qx = atof(xmlReader.readElementText().toStdString().c_str());

	token = xmlReader.readNext();//qy
	double qy = atof(xmlReader.readElementText().toStdString().c_str());

	token = xmlReader.readNext();//qz
	double qz = atof(xmlReader.readElementText().toStdString().c_str());

	token = xmlReader.readNext();//Rotation
	elementName = xmlReader.name().toString();

	Eigen::Matrix4d regpose = (Eigen::Affine3d(Eigen::Quaterniond(qw,qx,qy,qz))).matrix();
	regpose(0,3) = tx;
	regpose(1,3) = ty;
	regpose(2,3) = tz;

	return regpose;
}

pair<FrameVec, PoseVec> readViewXML(const string& roomLogName, const string& xmlFile)
{
    PoseVec poses;
    FrameVec frames;

	QFile file(xmlFile.c_str());
	if (!file.exists()){
		ROS_ERROR("Could not open file %s to load room.",xmlFile.c_str());
		exit(-1);
	}

	QString xmlFileQS(xmlFile.c_str());
	int index = xmlFileQS.lastIndexOf('/');
	string roomFolder = xmlFileQS.left(index).toStdString();

	file.open(QIODevice::ReadOnly);
	ROS_INFO_STREAM("Parsing xml file: "<<xmlFile.c_str());

	QXmlStreamReader xmlReader(&file);

	while (!xmlReader.atEnd() && !xmlReader.hasError()) {
		QXmlStreamReader::TokenType token = xmlReader.readNext();
		if (token == QXmlStreamReader::StartDocument) {
			continue;
        }

		if (xmlReader.hasError()) {
			ROS_ERROR("XML error: %s",xmlReader.errorString().toStdString().c_str());
			exit(-1);
		}

		QString elementName = xmlReader.name().toString();

		if (token == QXmlStreamReader::StartElement) {
			if (xmlReader.name() == "View") {
				cv::Mat rgb;
				cv::Mat depth;

				QXmlStreamAttributes attributes = xmlReader.attributes();
				if (attributes.hasAttribute("RGB")){
					string imgpath = attributes.value("RGB").toString().toStdString();
					rgb = cv::imread(roomFolder+"/"+imgpath, CV_LOAD_IMAGE_UNCHANGED);
				}
                else {
                    break;
                }

				if (attributes.hasAttribute("DEPTH")) {
					string imgpath = attributes.value("DEPTH").toString().toStdString();
					depth = cv::imread(roomFolder+"/"+imgpath, CV_LOAD_IMAGE_UNCHANGED);
				}
                else {
                    break;
                }


				token = xmlReader.readNext();//Stamp
				elementName = xmlReader.name().toString();

				token = xmlReader.readNext();//sec
				elementName = xmlReader.name().toString();
				int sec = atoi(xmlReader.readElementText().toStdString().c_str());

				token = xmlReader.readNext();//nsec
				elementName = xmlReader.name().toString();
				int nsec = atoi(xmlReader.readElementText().toStdString().c_str());
				token = xmlReader.readNext();//end stamp

				token = xmlReader.readNext();//Camera
				elementName = xmlReader.name().toString();

                Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
				//reglib::Camera * cam = new reglib::Camera();

				token = xmlReader.readNext();//fx
				//cam->fx = atof(xmlReader.readElementText().toStdString().c_str());
                K(0, 0) = atof(xmlReader.readElementText().toStdString().c_str());

				token = xmlReader.readNext();//fy
				//cam->fy = atof(xmlReader.readElementText().toStdString().c_str());
                K(1, 1) = atof(xmlReader.readElementText().toStdString().c_str());

				token = xmlReader.readNext();//cx
				//cam->cx = atof(xmlReader.readElementText().toStdString().c_str());
                K(0, 2) = atof(xmlReader.readElementText().toStdString().c_str());

				token = xmlReader.readNext();//cy
				//cam->cy = atof(xmlReader.readElementText().toStdString().c_str());
                K(1, 2) = atof(xmlReader.readElementText().toStdString().c_str());

				token = xmlReader.readNext();//Camera
				elementName = xmlReader.name().toString();

				double time = double(sec)+double(nsec)/double(1e9);

				token = xmlReader.readNext();//RegisteredPose
				elementName = xmlReader.name().toString();

				Eigen::Matrix4d regpose = getPose(xmlReader);

				token = xmlReader.readNext();//RegisteredPose
				elementName = xmlReader.name().toString();


				token = xmlReader.readNext();//Pose
				elementName = xmlReader.name().toString();

				Eigen::Matrix4d pose = getPose(xmlReader);

				token = xmlReader.readNext();//Pose
				elementName = xmlReader.name().toString();

                // this is what we need to replace, I don't want the rgbdframe type, let's have our own struct instead
				//reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,rgb,depth, time, regpose,true,savePath,compute_edges);
				//frame->keyval = roomLogName+"_frame_"+to_string(frames.size());

                SimpleFrame frame { rgb, depth, time, K, pose };

				frames.push_back(frame);
				poses.push_back(pose);
			}
		}
	}

    return make_pair(frames, poses);
}

pair<ObjectVec, FrameVec> loadObjects(const string& path)
{
	printf("loadModels(%s)\n",path.c_str());

	SimpleXMLParser<pcl::PointXYZRGB> parser;
	SimpleXMLParser<pcl::PointXYZRGB>::RoomData roomData  = parser.loadRoomFromXML(path, vector<string>(), false, false);
	string roomLogName = roomData.roomLogName; // is this the only place where roomData is used? seems unnecessary
    // but on the other hand, we need e.g. the time of the sweep and maybe some transforms to get to global coordinates
	printf("roomLogName: %s\n",roomLogName.c_str());

	int slash_pos = path.find_last_of("/");
	string sweep_folder = path.substr(0, slash_pos) + "/";

	FrameVec frames;
	PoseVec poses;
	tie(frames, poses) = readViewXML(roomLogName, sweep_folder+"ViewGroup.xml");

    ObjectVec objects;
	int objcounter = -1;
	QStringList objectFiles = QDir(sweep_folder.c_str()).entryList(QStringList("dynamic_obj*.xml"));
	for (auto objectFile : objectFiles) {
		objcounter++;
		string objectStr = sweep_folder+objectFile.toStdString();
		QFile file(objectStr.c_str());
		if (!file.exists()) {
            ROS_ERROR("Could not open file %s to masks.",objectStr.c_str());
            continue;
        }
		file.open(QIODevice::ReadOnly);

        SegmentedObject object;
		//reglib::Model * mod = new reglib::Model();
		//mod->keyval = roomLogName+"_object_"+to_string(objcounter);
		//printf("object label: %s\n",mod->keyval.c_str());

		QXmlStreamReader xmlReader(&file);

		while (!xmlReader.atEnd() && !xmlReader.hasError()) {
			QXmlStreamReader::TokenType token = xmlReader.readNext();
			if (token == QXmlStreamReader::StartDocument) {
				continue;
            }

			if (xmlReader.hasError()) {
				ROS_ERROR("XML error: %s",xmlReader.errorString().toStdString().c_str());
				break;
			}

			QString elementName = xmlReader.name().toString();
			if (token == QXmlStreamReader::StartElement) {
				if (xmlReader.name() == "Mask") {
					int number = 0;
					cv::Mat mask;
					QXmlStreamAttributes attributes = xmlReader.attributes();

					if (attributes.hasAttribute("filename")) {
						QString maskpath = attributes.value("filename").toString();
						mask = cv::imread(sweep_folder+"/"+(maskpath.toStdString().c_str()), CV_LOAD_IMAGE_UNCHANGED);
					}
                    else {
                        break;
                    }

					if (attributes.hasAttribute("image_number")) {
						QString depthpath = attributes.value("image_number").toString();
						number = atoi(depthpath.toStdString().c_str());
					}
                    else {
                        break;
                    }

					//mod->frames.push_back(frames[number]->clone());
					//mod->relativeposes.push_back(poses[number]);
					//mod->modelmasks.push_back(new reglib::ModelMask(mask));
                    object.frames.push_back(number);
                    object.masks.push_back(mask);
                    object.relative_poses.push_back(poses[number]);
				}
			}
		}

		//models.push_back(mod);
        objects.push_back(object);
	}

	return make_pair(objects, frames);
}

void save_objects(const string& room_path, ObjectVec& objects, FrameVec& frames)
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

    pair<ObjectVec, FrameVec> objects = loadObjects(room_path);
    save_objects(room_path, objects.first, objects.second);

    return 0;
}
