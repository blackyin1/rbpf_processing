
using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using SweepT = semantic_map_load_utilties::IntermediateCloudCompleteData<PointT>;

void propagate_changes(const string& sweep_xml, bool backwards)
{
    PoseVec current_transforms;
    string previous_xml;
    tie(previous_xml, current_transforms) = read_previous_sweep_params(sweep_xml, backwards);

    PoseVec previous_transforms = load_transforms_for_data(previous_data);

    ObjectVec previous_objects;
    FrameVec previous_frames;
    tie(previous_objects, previous_frames) = loadObjects(previous_xml);

    ObjectVec current_objects;
    FrameVec current_frames;
    tie(current_objects, current_frames) = loadObjects(sweep_xml);

    // some premature optimization....
    PoseVec Mp;
    for (size_t i = 0; i < previous_frames.size(); ++i) {
        Eigen::Matri4xd Kp = Eigen::Matrix4d::Identity();
        Kp.topLeftCorner<3, 3>() = previous_frames[i].K;
        Eigen::Matrix4d Mpi = Kp.transpose().colPivHouseholderQr().solve(previous_transforms[i]).transpose();
        Mp.push_back(Mpi);
    }

    PoseVec Mc;
    for (size_t i = 0; i < current_frames.size(); ++i) {
        Eigen::Matri4xd Kc = Eigen::Matrix4d::Identity();
        Kc.topLeftCorner<3, 3>() = current_frames[i].K;
        Eigen::Matrix4d Mci = current_transforms[i].transpose().colPivHouseholderQr().solve(Kc).transpose();
        Mc.push_back(Mci);
    }

    for (SegmentedObject& obj : previous_objects) {
        // project this into every frame of the current sweep, look at the static areas
        // ok, let's keep it like that for now, if it's static it should be propagated
        // but it could also be gone, so even there, we need to make sure that the depth values
        // are not much larger than they should be
        // also, backwards change detection could detect it as dynamic if it disappears
        // BUT: obj contains multiple masks, one for every image it's in
        // should I project all of these images? would require depth for that
        // also, clearly need the camera parameters here ;)

        // assumption: one object here can only correspond to one object in new frame
        for (size_t i = 0; i < obj.frames.size(); ++i) {

            size_t frame_ind = obj.frames[i];
            Eigen::Matrix4d relative_pose = previous_transforms[frame_ind];

            Eigen::Matrix<double, 4, Eigen::Dynamic> Dp;
            cv::Mat locations;   // output, locations of non-zero pixels
            cv::findNonZero(obj.masks[i], locations);
            for (size_t j = 0; j < locations.cols; ++j) {
                int x = locations.at<int>(0, j);
                int y = locations.at<int>(1, j);
                double depth = double(previous_frames[i].depth.at<uint16_t>(y, x))/1000.0; // should this be 500?
                Dp.col(j) << depth*x, depth*y, depth, 1.0;
            }

            for (size_t j = 0; j < current_frames.size(); ++j) {
                Eigen::Matrix<double, 4, Eigen::Dynamic> Dc = Mc[j]*Mp[i]*Dp;
                Dc.topRows<2>() = Dc.topRows<2>().array().rowwise() / Dc.row(2).array();
                Dc.row(2) = Dc.row(2).array()/Dc.row(3).array();
                Eigen::Matrix<int, 4, Eigen::Dynamic> pc = Dc.topRows<2>().cast<int>();
                cv::Mat mask = cv::Mat::zeros(CV_8UC, 480, 640);
                cv::Mat depth = cv::Mat::zeros(CV_16UC1, 480, 640);
                size_t pixel_counter = 0;
                for (size_t k = 0; k < pc.cols(); ++k) {
                    if (pc(0, k) >= 0 && pc(0, k) < 640 && pc(1, k) >= 0 && pc(1, k) < 480) {
                        mask.at<uchar_t>(pc(1, k), pc(0, k)) = 255;
                        depth.at<uint16_t>(pc(1, k), pc(0, k)) = uint16_t(1000.0*Dc(2, k)); // should this be 500?
                        ++pixel_counter;
                    }
                }

                // maybe I should simply check if it's still here or not, and if occluded
                // afterwards I can check if there is any overlap with current dynamic objects
                // or with the backward dynamic objects. In that case I can just skip
                // I should probably save all of the relevant objects at the same time as
                // doing this pass
            }
        }
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

    string room_path(argv[1]);
    propagate_changes(sweep_xml, backwards);

    return 0;
}
