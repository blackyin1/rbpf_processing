#ifndef XML_CONVENIENCE_H
#define XML_CONVENIENCE_H

#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <rbpf_processing/data_convenience.h>

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using RoomT = SimpleXMLParser<PointT>::RoomData;
using FrameVec = vector<SimpleFrame>;

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

    cout << "Got transforms to previous sweep..." << endl;

    return make_pair(previous_sweep_xml, sweep_transforms);
}

/*
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
*/

/*
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
*/

pair<FrameVec, PoseVec> load_frames_poses(RoomT& data)
{
    //RoomT data = SimpleXMLParser<PointT>::loadRoomFromXML(sweep_xml, vector<string>{"RoomIntermediateCloud"}, false, false);
    FrameVec frames;
    PoseVec poses;

    cout << "Got room with " << data.vIntermediateRoomClouds.size() << " number of clouds..." << endl;

    for (size_t i = 0; i < data.vIntermediateRoomClouds.size(); ++i) {
        //cout << "Loading pose and K: " << i << endl;
        Eigen::Affine3d e;
        tf::transformTFToEigen(data.vIntermediateRoomCloudTransformsRegistered[i], e);
        poses.push_back(e.matrix());
        //pair<cv::Mat, cv::Mat> images = SimpleXMLParser<PointT>::createRGBandDepthFromPC(data.vIntermediateRoomClouds[i]);
        SimpleFrame frame;
        frame.rgb = data.vIntermediateRGBImages[i]; // images.first;
        frame.depth = data.vIntermediateDepthImages[i]; // images.second;
        //cout << "Depth image size: " << frame.depth.rows << "x" << frame.depth.cols << endl;
        frame.pose = e.matrix();
        image_geometry::PinholeCameraModel model = data.vIntermediateRoomCloudCamParams[0];
        cv::Matx33d cvK = model.intrinsicMatrix();
        frame.K = Eigen::Map<Eigen::Matrix3d>(cvK.val).transpose();
        frames.push_back(frame);
    }

    return make_pair(frames, poses);
}

pair<ObjectVec, FrameVec> loadObjects(const string& path, bool backwards = false)
{
    printf("loadModels(%s)\n",path.c_str());

    SimpleXMLParser<pcl::PointXYZRGB> parser;
    SimpleXMLParser<pcl::PointXYZRGB>::RoomData roomData  = parser.loadRoomFromXML(path, vector<string>{"RoomIntermediateCloud"}, false, true);
    string roomLogName = roomData.roomLogName; // is this the only place where roomData is used? seems unnecessary
    // but on the other hand, we need e.g. the time of the sweep and maybe some transforms to get to global coordinates
    printf("roomLogName: %s\n",roomLogName.c_str());

    int slash_pos = path.find_last_of("/");
    string sweep_folder = path.substr(0, slash_pos) + "/";

    cout << "Getting poses for " << path << endl;

    FrameVec frames;
    PoseVec poses;
    //tie(frames, poses) = readViewXML(roomLogName, sweep_folder+"ViewGroup.xml"); // this does not seem to work but why use this if we have the metaroom parser?
    tie(frames, poses) = load_frames_poses(roomData);

    ObjectVec objects;
    int objcounter = -1;
    QStringList objectFiles;
    if (backwards) {
        objectFiles = QDir(sweep_folder.c_str()).entryList(QStringList("back_dynamic_obj*.xml"));
    }
    else {
        objectFiles = QDir(sweep_folder.c_str()).entryList(QStringList("dynamic_obj*.xml"));
    }
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
        object.object_type = "detected";
        object.going_backward = backwards;
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

    cout << "Done loading objects for " << path << endl;

    return make_pair(objects, frames);
}

#endif // XML_CONVENIENCE_H
