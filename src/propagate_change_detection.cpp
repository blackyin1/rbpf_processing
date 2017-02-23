
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

            cv::Mat locations;   // output, locations of non-zero pixels
            cv::findNonZero(obj.masks[i], locations);
            for (size_t j = 0; j < locations.cols; ++j) {

            }

            for (size_t j = 0; j < current_frames.size(); ++j) {
                
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
