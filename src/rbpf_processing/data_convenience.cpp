#include <rbpf_processing/data_convenience.h>

using namespace std;

void add_cropped_rgb_to_object(SegmentedObject& obj, FrameVec& frames)
{
    obj.cropped_rgbs.clear();
    for (size_t i = 0; i < obj.frames.size(); ++i) {
        cv::Mat points;
        cv::findNonZero(obj.masks[i], points);
        cv::Rect min_rect = cv::boundingRect(points);
        cv::Mat cropped = frames[obj.frames[i]].rgb(min_rect);
        obj.cropped_rgbs.push_back(cropped);
    }
}

void add_pos_to_object(SegmentedObject& obj, FrameVec& frames, const Eigen::Matrix4d& map_pose)
{
    double scaling = 1000.0;

    double max_pix = 0;
    size_t max_ind = 0;
    for (size_t i = 0; i < obj.frames.size(); ++i) {
        double pix = cv::sum(obj.masks[i])[0]/255.0;
        if (pix > max_pix) {
            max_pix = pix;
            max_ind = i;
        }
    }

    cv::Mat points;
    cv::findNonZero(obj.masks[max_ind], points);
    cv::Mat depth = frames[obj.frames[max_ind]].depth;
    cv::Mat mask;
    cv::bitwise_and(obj.masks[max_ind], depth > 0, mask);

    double mean_depth = 1.0/scaling*cv::mean(depth, mask)[0];

    Eigen::Vector4d mean_vec;
    mean_vec(3) = 1.0;
    mean_vec(2) = 1.0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (size_t i = 0; i < points.total(); ++i) {
        cv::Point p = points.at<cv::Point>(i);
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_vec(0) = mean_x / double(points.total());
    mean_vec(1) = mean_y / double(points.total());

    mean_vec.head<3>() = frames[obj.frames[max_ind]].K.inverse()*mean_vec.head<3>();
    mean_vec.head<3>() *= mean_depth/mean_vec(2);

    mean_vec = map_pose*frames[obj.frames[max_ind]].pose*mean_vec;

    obj.pos = 1.0/mean_vec(3)*mean_vec.head<3>();
}

PoseVec load_transforms_for_data(SweepT& data)
{
    PoseVec transforms;
    for (tf::StampedTransform t : data.vIntermediateRoomCloudTransformsRegistered) {
        Eigen::Affine3d e;
        tf::transformTFToEigen(t, e);
        transforms.push_back(e.matrix());
    }
    return transforms;
}

CloudT::Ptr save_object_cloud(SegmentedObject& obj, FrameVec& frames,
                              const Eigen::Matrix4d& map_pose,
                              const string& object_path)
{
    double scaling = 1000.0;

    /*
    double max_pix = 0;
    size_t max_ind = 0;
    for (size_t i = 0; i < obj.frames.size(); ++i) {
        double pix = cv::sum(obj.masks[i])[0]/255.0;
        if (pix > max_pix) {
            max_pix = pix;
            max_ind = i;
        }
    }

    cv::Mat locations;   // output, locations of non-zero pixels
    cv::findNonZero(obj.masks[max_ind], locations);
    Eigen::Matrix<double, 4, Eigen::Dynamic> Dp(4, locations.total());
    Eigen::Matrix<int, 3, Eigen::Dynamic> Pp(3, locations.total());
    Dp.row(2).setOnes();

    cout << "Found nonzero with size " << locations.rows << "x" << locations.cols << ", type: " << locations.type() << endl;
    */

    CloudT::Ptr cloud(new CloudT);

    cout << "Saving object with " << obj.frames.size() << " frames..." << endl;

    for (size_t i = 0; i < obj.frames.size(); ++i) {

        cv::Mat locations;   // output, locations of non-zero pixels
        cv::findNonZero(obj.masks[i], locations);
        Eigen::Matrix<double, 4, Eigen::Dynamic> Dp(4, locations.total());
        Eigen::Matrix<int, 3, Eigen::Dynamic> Pp(3, locations.total());
        Dp.row(2).setOnes();

        Eigen::Matrix3d Kinv = frames[obj.frames[i]].K.inverse();
        for (size_t j = 0; j < locations.total(); ++j) {
            cv::Point p = locations.at<cv::Point>(j);
            cv::Vec3b c = frames[obj.frames[i]].rgb.at<cv::Vec3b>(p.y, p.x);
            Pp.block<3, 1>(0, j) << c[2], c[1], c[0];
            Dp.block<2, 1>(0, j) << p.x, p.y;
            Dp(3, j) = double(frames[obj.frames[i]].depth.at<uint16_t>(p.y, p.x))/scaling;
        }

        Dp.topRows<3>() = Kinv*Dp.topRows<3>();
        Dp.topRows<3>() = Dp.topRows<3>().array().rowwise() * (Dp.row(3).array() / Dp.row(2).array());
        Dp.row(3).setOnes();
        Dp = map_pose*frames[obj.frames[i]].pose*Dp;
        Dp.topRows<3>() = Dp.topRows<3>().array().rowwise() / Dp.row(3).array();

        for (size_t j = 0; j < locations.total(); ++j) {
            PointT p; p.getVector3fMap() = Dp.block<3, 1>(0, j).cast<float>();
            p.r = Pp(0, j); p.g = Pp(1, j); p.b = Pp(2, j);
            cloud->points.push_back(p);
        }

    }

    cout << "Object cloud size: " << cloud->size() << endl;

    pcl::io::savePCDFileBinary((PathT(object_path) / "cloud.pcd").string(), *cloud);

    return cloud;
}

void save_complete_propagated_cloud(vector<CloudT::Ptr>& clouds, const string& sweep_xml, bool backwards)
{
    cout << "Writing complete propagated cloud..." << endl;

    CloudT complete_cloud;

    for (CloudT::Ptr& cloud : clouds) {
        complete_cloud += *cloud;
    }

    cout << "Complete cloud size: " << complete_cloud.size() << endl;

    PathT complete_path;
    if (backwards) {
        complete_path = PathT(sweep_xml).parent_path() / "back_propagated_dynamic_clusters.pcd";
    }
    else {
        complete_path = PathT(sweep_xml).parent_path() / "propagated_dynamic_clusters.pcd";
    }

    pcl::io::savePCDFileBinary(complete_path.string(), complete_cloud);
}

void save_objects(ObjectVec& objects, FrameVec& frames, const Eigen::Matrix4d& map_pose,
                  const string& sweep_xml, bool backwards, bool save_cloud)
{
    if (objects.empty()) {
        return;
    }

    cout << "Saving objects, creating directory..." << endl;

    PathT objects_path = PathT(sweep_xml).parent_path() / "consolidated_objects";
    if (!boost::filesystem::exists(objects_path)) {
        boost::filesystem::create_directory(objects_path);
    }

    cout << "Creating object subdirectories..." << endl;

    // how many objects are already saved in this folder?
    size_t i = 0;
    while (true) {
        PathT object_path = objects_path / (string("object") + num_str(i));
        if (!boost::filesystem::exists(object_path)) {
            break;
        }
        ++i;
    }

    vector<CloudT::Ptr> clouds;

    // ok, save this in a format that we can use to extract the CNN features (JPEG FTW)
    // one thing to note: we'll have to do another pass where we get all the image paths
    // fortunately, there's python
    for (SegmentedObject& obj : objects) {
        cout << "Saving object " << i << endl;
        // let's create a folder for every object
        PathT object_path = objects_path / (string("object") + num_str(i));
        boost::filesystem::create_directory(object_path);
        obj.object_folder = object_path.string();
        PathT object_file = object_path / "segmented_object.json";
        cout << "Adding rgb images..." << endl;
        add_cropped_rgb_to_object(obj, frames);
        cout << "Adding pos to object..." << endl;
        add_pos_to_object(obj, frames, map_pose);
        cout << "Writing object cloud..." << endl;
        CloudT::Ptr cloud = save_object_cloud(obj, frames, map_pose, object_path.string());
        if (obj.object_type == "propagated") {
            clouds.push_back(cloud);
        }
        cout << "Writing object..." << endl;
        ofstream out(object_file.string());
        {
            cereal::JSONOutputArchive archive_o(out);
            archive_o(cereal::make_nvp("object", obj));
        }
        ++i;
    }

    if (!clouds.empty() && save_cloud) {
        save_complete_propagated_cloud(clouds, sweep_xml, backwards);
    }

    cout << "Done saving objects..." << endl;
}

void save_complete_objects(ObjectVec& objects, FrameVec& frames,
                           const Eigen::Matrix4d& map_pose, const string& sweep_xml)
{
    if (objects.empty()) {
        return;
    }

    cout << "Saving objects, creating directory..." << endl;

    PathT objects_path = PathT(sweep_xml).parent_path() / "consolidated_objects";
    if (boost::filesystem::exists(objects_path)) {
        boost::filesystem::remove_all(objects_path);
    }
    boost::filesystem::create_directory(objects_path);

    cout << "Creating object subdirectories..." << endl;

    // how many objects are already saved in this folder?
    size_t i = 0;
    while (true) {
        PathT object_path = objects_path / (string("object") + num_str(i));
        if (!boost::filesystem::exists(object_path)) {
            break;
        }
        ++i;
    }

    // ok, save this in a format that we can use to extract the CNN features (JPEG FTW)
    // one thing to note: we'll have to do another pass where we get all the image paths
    // fortunately, there's python
    for (SegmentedObject& obj : objects) {
        cout << "Saving object " << i << endl;
        // let's create a folder for every object
        PathT object_path = objects_path / (string("object") + num_str(i));
        boost::filesystem::create_directory(object_path);
        obj.object_folder = object_path.string();
        PathT object_file = object_path / "segmented_object.json";
        cout << "Adding rgb images..." << endl;
        add_cropped_rgb_to_object(obj, frames);
        cout << "Adding pos to object..." << endl;
        add_pos_to_object(obj, frames, map_pose);
        cout << "Writing object cloud..." << endl;
        save_object_cloud(obj, frames, map_pose, object_path.string());
        cout << "Writing object..." << endl;
        ofstream out(object_file.string());
        {
            cereal::JSONOutputArchive archive_o(out);
            archive_o(cereal::make_nvp("object", obj));
        }
        ++i;
    }

    cout << "Done saving objects..." << endl;
}

ObjectVec load_propagated_objects(const string& sweep_xml, bool do_filter, bool backwards)
{
    ObjectVec objects;

    PathT objects_path = PathT(sweep_xml).parent_path() / "consolidated_objects";
    if (!boost::filesystem::exists(objects_path)) {
        return objects;
    }


    for (size_t i = 0; ; ++i) {
        PathT object_path = objects_path / (string("object") + num_str(i));
        if (!boost::filesystem::exists(object_path)) {
            break;
        }
        PathT object_file = object_path / "segmented_object.json";
        SegmentedObject obj;
        ifstream in(object_file.string());
        {
            cereal::JSONInputArchive archive_i(in);
            archive_i(obj);
        }
        if (!do_filter || obj.going_backward == backwards) {
            cout << "LOADING PROPAGATED OBJECT!" << endl;
            objects.push_back(obj);
        }
    }

    return objects;
}
