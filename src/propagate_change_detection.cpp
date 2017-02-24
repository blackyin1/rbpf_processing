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
#include <rbpf_processing/data_convenience.h>

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using SweepT = semantic_map_load_utilties::IntermediateCloudCompleteData<PointT>;

ObjectVec project_objects(const PoseVec& current_transforms, const PoseVec& previous_transforms,
                          FrameVec& current_frames, FrameVec& previous_frames, ObjectVec& previous_objects)
{
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

    ObjectVec projected_objects;
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

            vector<cv::Mat> projected_masks;
            vector<cv::Mat> projected_depths;
            vector<size_t> frame_ids;
            for (size_t j = 0; j < current_frames.size(); ++j) {

                // short wire here if there is no overlap between frames. Best way is probably to check
                // angle between z vectors projected in x-y plane. If > 90 degress, break
                Eigen::Vector3d d1 = previous_transforms[frame_ind].col(2);
                Eigen::Vector3d d2 = current_transforms[i].col(2);
                double angle = atan2(d1.cross(d2).norm(), d1.dot(d2));
                if (angle > 0.5*M_PI) {
                    continue;
                }

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

                if (pixel_counter > 1000) { // this is an arbitrary threshold
                    projected_masks.push_back(mask);
                    projected_depths.push_back(depth);
                    frame_ids.push_back(j);
                }
            }

            if (!projected_masks.empty()) {
                SegmentedObject obj;
                obj.frames.insert(obj.frames.end(), frame_ids.begin(), frame_ids.end());
                obj.masks.insert(obj.masks.end(), projected_masks.begin(), projected_masks.end());
                obj.depth_masks.insert(obj.depth_masks.end(), projected_depths.begin(), projected_depths.end());
            }
        }
    }

    return projected_objects;
}

// this could be useful also for comparing forward and backward computed objects
// also note that if we have an overlap here, it should be with a backward/forward object, otherwise it's strange
ObjectVec propagate_objects(FrameVec& current_frames, ObjectVec& projected_objects)
{
    // how many frames should I check here? all of them!
    ObjectVec new_objects;

    for (SegmentedObject& obj : projected_objects) {

        double absdiff = 0;
        double totsum = 0;
        double totpixels = 0;

        for (size_t i = 0; i < obj.frames.size(); ++i) {
            // it would probably be good to extract just the cropped depth
            cv::Mat bg = current_frames[obj.frames[i]].depth;
            bg.setTo(0, obj.masks[i]);
            bg.convertTo(resized_image, CV_32FC1);
            cv::Mat proj = obj.depth_masks[i];
            proj.convertTo(resized_image, CV_32FC1);

            cv::Mat diff = proj - bg; // we should probably change this to SC32?
            absdiff += cv::sum(cv::abs(diff));
            totsum += cv::sum(diff);
            totpixels += cv::sum(obj.masks[i]);

            // 3 interesting cases: gone (positive difference), occluded (negative difference), present (very small difference)
            // one thing that just struck me is that objects will probably be occluded by themselves quite often
        }

        absdiff /= totpixels;
        totsum /= totpixels;

        if (absdiff < 0.03) { // it's probably there still
            SegmentedObject propagated;
            propagated.frames = obj.frames;
            propagated.relative_poses = obj.relative_poses;
            for (size_t i = 0; i < obj.frames.size(); ++i) {
                propagated.masks.push_back(obj.masks[i].clone());
            }
            new_objects.push_back(propagated);
        }
        else if (totsum > 0.1) { // it's probably occluded
            // the question is, what should we do in this case?
            // let's propagate it, but keep a flag indicating occlusion
            // btw, wouldn't we need to also propagate depth in that case?
            // so, it seems the best thing would be to actually propagate
            // the segmentation from the last scene rather than this one
            /*
            SegmentedObject propagated;
            propagated.frames = obj.frames;
            propagated.relative_poses = obj.relative_poses;
            for (size_t i = 0; i < obj.frames.size(); ++i) {
                propagated.masks.push_back(obj.masks[i].clone());
            }
            new_objects.push_back(propagated);
            */
        }
        else {
            // it seems like it's not there, do nothing
        }

    }

    return new_objects;
}

ObjectVec filter_objects(ObjectVec& objects, ObjectVec& filter_by)
{
    ObjectVec filtered_objects;

    for (SegmentedObject& obj : objects) {
        for (SegmentedObject& prev : filter_by) {

            double total_overlap = 0;
            double total_pixels = 0;

            // these share frames, let's step through frames in parallell
            size_t i = 0; size_t j = 0;
            while (i < obj.frames.size() && j < prev.frames.size()) {
                if (obj.frames[i] < prev.frames[j]) {
                    ++i;
                    continue;
                }
                else if (obj.frames[i] > prev.frames[j]) {
                    ++j;
                    continue;
                }
                // obj.frames[i] == prev.frames[j]
                cv::Mat overlap;
                cv::bitwise_and(obj.masks[i], prev.masks[j], overlap);
                cv::Mat total;
                cv::bitwise_and(obj.masks[i], prev.masks[j], total);

                total_overlap += cv::sum(overlap);
                total_pixels += cv::sum(total);
            }

            if (total_pixels == 0 || total_overlap / total_pixels < 0.5) { // pretty liberal but: it's not the same!
                SegmentedObject filtered;
                filtered.frames = obj.frames;
                filtered.relative_poses = obj.relative_poses;
                for (size_t i = 0; i < obj.frames.size(); ++i) {
                    filtered.masks.push_back(obj.masks[i].clone());
                }
                filtered_objects.push_back(filtered);
            }

        }
    }

    return filtered_objects;
}

void save_objects(ObjectVec& objects, FrameVec& frames, const string& sweep_xml, bool backwards)
{
    PathT objects_path = PathT(sweep_xml).parent_path();
    if (!boost::filesystem::exists(objects_path)) {
        boost::filesystem::create_directory(object_path);
    }

    // how many objects are already saved in this folder?
    size_t i = 0;
    while (true) {
        PathT object_path = object_path / (string("object") + num_str(i));
        if (!boost::filesystem::exists(objects_path)) {
            break;
        }
    }

    // ok, save this in a format that we can use to extract the CNN features (JPEG FTW)
    // one thing to note: we'll have to do another pass where we get all the image paths
    // fortunately, there's python
    for (SegmentedObject& obj : objects) {
        // let's create a folder for every object
        PathT object_path = object_path / (string("object") + num_str(i));
        boost::filesystem::create_directory(object_path);
        obj.object_folder = object_path.string();
        PathT object_file = object_path / "segmented_object.json";
        add_cropped_rgb_to_object(obj, frames);
        ofstream out(object_file.string());
        {
            cereal::JSONOutputArchive archive_o(out);
            archive_o(obj);
        }
        ++i;
    }
}

void propagate_changes(const string& sweep_xml, bool backwards)
{
    PoseVec current_transforms;
    string previous_xml;
    tie(previous_xml, current_transforms) = read_previous_sweep_params(sweep_xml, backwards);

    PoseVec previous_transforms = load_transforms_for_data(previous_data);

    ObjectVec previous_objects;
    FrameVec previous_frames;
    // note that these objects should also, eventually include the ones that have been propagated forwards
    tie(previous_objects, previous_frames) = loadObjects(previous_xml, backwards);

    ObjectVec current_objects;
    FrameVec current_frames;
    // note that these objects should also, eventually include the ones that have been propagated backwards
    tie(current_objects, current_frames) = loadObjects(sweep_xml, !backwards); // we are interested in the objects coming from the other direction

    ObjectVec projected_objects = project_objects(current_transforms, previous_transforms, current_frames,
                                                  previous_frames, previous_objects);

    // maybe I should simply check if it's still here or not, and if occluded
    // afterwards I can check if there is any overlap with current dynamic objects
    // or with the backward dynamic objects. In that case I can just skip
    // I should probably save all of the relevant objects at the same time as
    // doing this pass
    ObjectVec propagated_objects = propagate_objects(current_objects, current_frames, projected_objects);

    ObjectVec filtered_objects = filter_objects(propagated_objects, current_objects);

    // save forwards and backwards objects except for the ones that overlap and the forward filtered objects
    save_objects(filtered_objects, current_frames, sweep_xml, backwards);
    save_objects(current_objects, current_frames, sweep_xml, !backwards);
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
