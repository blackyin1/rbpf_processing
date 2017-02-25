#include <eigen3/Eigen/Dense>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <metaroom_xml_parser/simple_summary_parser.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <tf_conversions/tf_eigen.h>
#include <rbpf_processing/data_convenience.h>
#include <rbpf_processing/xml_convenience.h>

using namespace std;

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;
using PathT = boost::filesystem::path;
using PoseVec = vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >;
using SweepT = semantic_map_load_utilties::IntermediateCloudCompleteData<PointT>;

ObjectVec project_objects(const PoseVec& current_transforms, const PoseVec& previous_transforms,
                          FrameVec& current_frames, FrameVec& previous_frames, ObjectVec& previous_objects)
{
    cout << "Projecting objects..." << endl;

    double scaling = 1000.0;

    // some premature optimization....
    PoseVec Mp;
    for (size_t i = 0; i < previous_frames.size(); ++i) {
        Eigen::Matrix4d Kp = Eigen::Matrix4d::Identity();
        Kp.topLeftCorner<3, 3>() = previous_frames[i].K;
        Eigen::Matrix4d Mpi = Kp.transpose().colPivHouseholderQr().solve(previous_transforms[i]).transpose();
        //Eigen::Matrix4d Mpi = previous_transforms[i].inverse()*Kp.inverse();
        Mp.push_back(Mpi);
        cout << "Mpi: \n" << Mpi << endl;
        cout << "Kp: " << Kp << endl;
    }

    PoseVec Mc;
    for (size_t i = 0; i < current_frames.size(); ++i) {
        Eigen::Matrix4d Kc = Eigen::Matrix4d::Identity();
        Kc.topLeftCorner<3, 3>() = current_frames[i].K;
        Eigen::Matrix4d Mci = current_transforms[i].transpose().colPivHouseholderQr().solve(Kc).transpose();
        //Eigen::Matrix4d Mci = Kc*current_transforms[i];
        Mc.push_back(Mci);
        cout << "Mci: \n" << Mci << endl;
        cout << "Kc: " << Kc << endl;
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

            cout << "Going through previous object " << i << endl;

            size_t frame_ind = obj.frames[i];
            //Eigen::Matrix4d relative_pose = previous_transforms[frame_ind];


            cv::Mat locations;   // output, locations of non-zero pixels
            cv::findNonZero(obj.masks[i], locations);
            Eigen::Matrix<double, 4, Eigen::Dynamic> Dp(4, locations.total());
            Dp.row(3).setOnes();

            cout << "Found nonzero with size " << locations.rows << "x" << locations.cols << ", type: " << locations.type() << endl;

            Eigen::Matrix3d Kpinv = previous_frames[frame_ind].K.inverse();
            for (size_t j = 0; j < locations.total(); ++j) {
                //int x = locations.at<int>(0, j);
                //int y = locations.at<int>(1, j);
                cv::Point p = locations.at<cv::Point>(j);
                int x = p.x;
                int y = p.y;
                //cout << "Previous frames size: " << previous_frames.size() << ", frame_ind: " << frame_ind << endl;
                //cout << previous_frames[frame_ind].depth.rows << ", " << previous_frames[frame_ind].depth.cols << ", " << y << ", " << x << endl;
                double depth = double(previous_frames[frame_ind].depth.at<uint16_t>(y, x))/scaling; // should this be 500?
                //Dp.col(j) = Eigen::Vector4d(depth*x, depth*y, depth, 1.0);
                Dp.block<3, 1>(0, j) = Kpinv*Eigen::Vector3d(double(x), double(y), 1.0);
                Dp.block<3, 1>(0, j) *= depth/Dp(2, j);
            }

            cout << "Done computing points..." << endl;

            vector<cv::Mat> projected_masks;
            vector<cv::Mat> projected_depths;
            vector<size_t> frame_ids;
            for (size_t j = 0; j < current_frames.size(); ++j) {

                //cout << "Going through current frame " << j << endl;

                // short wire here if there is no overlap between frames. Best way is probably to check
                // angle between z vectors projected in x-y plane. If > 90 degress, break
                Eigen::Vector3d d1 = previous_transforms[frame_ind].block<3, 1>(0, 2);
                Eigen::Vector3d d2 = current_transforms[j].block<3, 1>(0, 2);
                double angle = atan2(d1.cross(d2).norm(), d1.dot(d2));
                if (angle > 0.5*M_PI) {
                    continue;
                }

                //Eigen::Matrix<double, 4, Eigen::Dynamic> Dc = Mc[j]*Mp[i]*Dp;

                Eigen::Matrix<double, 4, Eigen::Dynamic> Dc = previous_transforms[i]*Dp;
                Dc.topRows<3>() = Dc.topRows<3>().array().rowwise() / Dc.row(3).array();
                Dc = current_transforms[j].inverse()*Dc;
                Dc.topRows<3>() = Dc.topRows<3>().array().rowwise() / Dc.row(3).array();
                Dc.topRows<3>() = current_frames[i].K*Dc.topRows<3>();
                Dc.topRows<2>() = Dc.topRows<2>().array().rowwise() / Dc.row(2).array();

                //Eigen::Matrix<double, 4, Eigen::Dynamic> Dc = Mc[j]*previous_transforms[i]*Dp;
                //Dc.topRows<2>() = Dc.topRows<2>().array().rowwise() / Dc.row(2).array(); //(Dc.row(2).array()*Dc.row(3).array());
                //Dc.row(2) = Dc.row(2).array()/Dc.row(3).array();
                Eigen::Matrix<int, 2, Eigen::Dynamic> pc = Dc.topRows<2>().cast<int>();
                cv::Mat mask = cv::Mat::zeros(480, 640, CV_8U);
                cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);
                size_t pixel_counter = 0;
                for (size_t k = 0; k < pc.cols(); ++k) {
                    if (pc(0, k) >= 0 && pc(0, k) < 640 && pc(1, k) >= 0 && pc(1, k) < 480) {
                        //cout << mask.rows << ", " << mask.cols << ", " << pc(1, k) << ", " << pc(0, k) << endl;
                        mask.at<uchar>(pc(1, k), pc(0, k)) = 255;
                        depth.at<uint16_t>(pc(1, k), pc(0, k)) = uint16_t(scaling*Dc(2, k)); // should this be 500?
                        ++pixel_counter;
                    }
                }

                if (pixel_counter > 1000) { // this is an arbitrary threshold
                    projected_masks.push_back(mask);
                    projected_depths.push_back(depth);
                    if (false) {
                        cv::imshow("original depth", 5*previous_frames[frame_ind].depth);
                        cv::imshow("projected depth", 5*depth);
                        cv::waitKey();
                    }
                    frame_ids.push_back(j);
                }
            }

            cout << "Found " << projected_masks.size() << " objects, converting..." << endl;

            if (!projected_masks.empty()) {
                SegmentedObject obj;
                obj.frames.insert(obj.frames.end(), frame_ids.begin(), frame_ids.end());
                obj.masks.insert(obj.masks.end(), projected_masks.begin(), projected_masks.end());
                obj.depth_masks.insert(obj.depth_masks.end(), projected_depths.begin(), projected_depths.end());
                projected_objects.push_back(obj);
            }
        }
    }

    cout << "Done projecting objects..." << endl;

    return projected_objects;
}

// this could be useful also for comparing forward and backward computed objects
// also note that if we have an overlap here, it should be with a backward/forward object, otherwise it's strange
ObjectVec propagate_objects(FrameVec& current_frames, ObjectVec& projected_objects)
{
    cout << "Propagating objects..." << endl;

    double scaling = 1000.0;

    // how many frames should I check here? all of them!
    ObjectVec new_objects;
    size_t occluded_objects = 0;

    size_t counter = 0;
    for (SegmentedObject& obj : projected_objects) {

        double absdiff = 0;
        double totsum = 0;
        double totpixels = 0;

        for (size_t i = 0; i < obj.frames.size(); ++i) {
            // it would probably be good to extract just the cropped depth
            // let's just keep the images as integers, we can convert to float after summing differences
            cv::Mat bg;
            current_frames[obj.frames[i]].depth.convertTo(bg, CV_32SC1);
            cv::Mat inverted_mask;
            cv::bitwise_not(obj.masks[i], inverted_mask);
            bg.setTo(0, inverted_mask);
            cv::Mat proj;
            obj.depth_masks[i].convertTo(proj, CV_32SC1);

            cout << "Current frames size: " << current_frames.size() << ", index: " << obj.frames[i] << endl;
            cout << "Whole background mean, raw: " << 1.0/scaling*cv::mean(current_frames[obj.frames[i]].depth, current_frames[obj.frames[i]].depth > 0)[0] << endl;
            cout << "Mean projected depth value: " << 255.0/scaling*cv::sum(proj)[0]/cv::sum(obj.masks[i])[0] << endl;
            cout << "Mean background depth value: " << 255.0/scaling*cv::sum(bg)[0]/cv::sum(obj.masks[i])[0] << endl;

            cv::Mat diff = proj - bg; // we should probably change this to SC32?
            cout << cv::sum(cv::abs(diff))[0] << endl;
            absdiff += 1.0/scaling*cv::sum(cv::abs(diff))[0];
            totsum += 1.0/scaling*cv::sum(diff)[0];
            totpixels += 1.0/255.0*cv::sum(obj.masks[i])[0];

            // 3 interesting cases: gone (positive difference), occluded (negative difference), present (very small difference)
            // one thing that just struck me is that objects will probably be occluded by themselves quite often
        }

        absdiff /= totpixels;
        totsum /= totpixels;

        cout << "Absdiff: " << absdiff << endl;
        cout << "Pixels: " << totpixels << endl;
        cout << "Totsum: " << totsum << endl;

        if (absdiff < 0.2) { // 0.03) { // it's probably there still
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
            ++occluded_objects;
        }
        else {
            // it seems like it's not there, do nothing
        }

        ++counter;
    }

    cout << "Got " << projected_objects.size() << " objects as input" << endl;
    cout << "Found " << new_objects.size() << " new objects and " << occluded_objects << " occluded ones " << endl;

    cout << "Done propagating objects..." << endl;

    return new_objects;
}

ObjectVec filter_objects(ObjectVec& objects, ObjectVec& filter_by)
{
    cout << "Filtering objects by comparing with previously detected..." << endl;

    ObjectVec filtered_objects;

    size_t counter = 0;
    for (SegmentedObject& obj : objects) {
        cout << "Checking if object " << counter << " is unique..." << endl;
        bool found = false;
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

                total_overlap += cv::sum(overlap)[0];
                total_pixels += cv::sum(total)[0];

                ++i;
                ++j;
            }

            if (total_pixels > 0 && total_overlap / total_pixels > 0.5) { // pretty liberal but: it's not the same!
                found = true;
                break;
            }

        }

        if (!found) {
            cout << "It's unique, adding!" << endl;
            SegmentedObject filtered;
            filtered.frames = obj.frames;
            filtered.relative_poses = obj.relative_poses;
            for (size_t i = 0; i < obj.frames.size(); ++i) {
                filtered.masks.push_back(obj.masks[i].clone());
            }
            filtered_objects.push_back(filtered);
        }
    }

    cout << "Done filtering objects, kept " << filtered_objects.size() << " out of " << objects.size() << endl;

    return filtered_objects;
}

void propagate_changes(const string& sweep_xml, bool backwards)
{
    PoseVec current_transforms;
    string previous_xml;
    tie(previous_xml, current_transforms) = read_previous_sweep_params(sweep_xml, backwards);

    SweepT previous_data = semantic_map_load_utilties::loadIntermediateCloudsCompleteDataFromSingleSweep<PointT>(previous_xml);
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
    ObjectVec propagated_objects = propagate_objects(current_frames, projected_objects);

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

    string sweep_xml(argv[1]);
    propagate_changes(sweep_xml, backwards);

    return 0;
}
