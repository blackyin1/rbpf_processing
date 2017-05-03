#include <rbpf_processing/object_processing.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(const std::vector<T>& vec,
                                          Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(const std::vector<T>& vec,
                                 const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](std::size_t i){ return vec[i]; });
    return sorted_vec;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> compute_object_cloud(SegmentedObject& obj, FrameVec& frames,
                                                              const Eigen::Matrix4d& map_pose)
{
    double scaling = 1000.0;

    Eigen::Matrix<double, 3, Eigen::Dynamic> complete_cloud;
    size_t nbr_points = 0;

    cout << "Merging object with " << obj.frames.size() << " frames..." << endl;

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

        complete_cloud.conservativeResize(3, nbr_points + locations.total());
        complete_cloud.rightCols(locations.total()) = Dp.topRows<3>().array().rowwise() / Dp.row(3).array();

        nbr_points += locations.total();
    }

    return complete_cloud;
}

// just merge two objects assuming they're from the same scan
void merge_objects(SegmentedObject& obj1, SegmentedObject& obj2)
{
    for (size_t i = 0; i < obj2.frames.size(); ++i) {
        auto iter = std::find(obj1.frames.begin(), obj1.frames.end(), obj2.frames[i]);
        if (iter == obj1.frames.end()) {
            obj1.frames.push_back(obj2.frames[i]);
            obj1.masks.push_back(obj2.masks[i].clone());
            if (obj2.depth_masks.size() > i) {
                obj1.depth_masks.push_back(obj2.depth_masks[i].clone());
            }
            if (obj2.cropped_rgbs.size() > i) {
                obj1.cropped_rgbs.push_back(obj2.cropped_rgbs[i].clone());
            }
        }
        else {
            size_t ind = std::distance(obj1.frames.begin(), iter);
            cv::Mat merged_mask;
            cv::bitwise_or(obj1.masks[ind], obj2.masks[i], merged_mask);
            obj1.masks[ind] = merged_mask;
            // too much work merging the depth masks and cropped rgb images also, not really needed
        }
    }

}

SegmentedObject clone_merge_objects(SegmentedObject& obj1, SegmentedObject& obj2)
{
    SegmentedObject obj;
    obj.going_backward = obj1.going_backward;
    obj.object_folder = obj1.object_folder;
    obj.object_type = obj1.object_type;
    obj.pos = obj1.pos;

    size_t i = 0; size_t j = 0;
    //while (i < obj1.frames.size() || j < obj2.frames.size()) {
    while (i < obj1.frames.size() && j < obj2.frames.size()) {
        if (j >= obj2.frames.size() || obj1.frames[i] < obj2.frames[j]) {
        //if (obj1.frames[i] < obj2.frames[j]) {
            obj.frames.push_back(obj1.frames[i]);
            obj.masks.push_back(obj1.masks[i].clone());
            cv::imshow("mask 1", obj1.masks.back());
            cv::waitKey();
            ++i;
            continue;
        }
        else if (i >= obj1.frames.size() || obj1.frames[i] > obj2.frames[j]) {
        //else if (obj1.frames[i] > obj2.frames[j]) {
            obj.frames.push_back(obj2.frames[j]);
            obj.masks.push_back(obj2.masks[j].clone());
            cv::imshow("mask 2", obj2.masks.back());
            cv::waitKey();
            ++j;
            continue;
        }

        cv::Mat all;
        cv::bitwise_or(obj1.masks[i], obj2.masks[j], all);

        obj.frames.push_back(obj1.frames[i]);
        obj.masks.push_back(all);

        ++i;
        ++j;
    }

    return obj;
}

void grow_hull(vector<cv::Point2f>& contour, float growth)
{
    cv::Moments m = cv::moments(contour, false);
    cv::Point2f c(m.m10 / m.m00, m.m01 / m.m00);
    for (cv::Point2f& p : contour) {
        float n = cv::norm(p - c);
        p += growth/n*(p - c);
    }
}

vector<cv::Point2f> compute_hull(vector<cv::Point2f>& points, float growth)
{
    vector<cv::Point2f> hull;
    cv::convexHull(points, hull, false);
    grow_hull(hull, growth);
    return hull;
}

template <typename T>
void remove_row(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

template <typename T>
void remove_column(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

template <typename T>
void apply_permuation_matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, const vector<std::size_t>& p)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp(matrix.rows(), matrix.cols());
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            temp(i, j) = matrix(p[i], p[j]);
        }
    }
    matrix = temp;
}

void merge_adjacencies(Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& adjacent, size_t i, size_t j)
{
    cout << "Merging " << i << " and " << j << endl;
    cout << adjacent << endl;

    // we want to delete j and merge it into i, merging with or statements
    for (size_t k = 0; k < adjacent.rows(); ++k) {
        adjacent(i, k) = adjacent(i, k) || adjacent(j, k);
        adjacent(k, i) = adjacent(i, k);
    }

    remove_row(adjacent, j);
    remove_column(adjacent, j);

    cout << adjacent << endl;
}

void consolidate_objects(ObjectVec& objects, vector<vector<cv::Point2f> >& hulls,
                         vector<vector<cv::Point2f> >& clouds, Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& adjacent,
                         float hull_grow, float fraction_smaller, float fraction_larger, vector<vector<size_t> >& consolidated_indices)
{
    size_t start_number_objects = objects.size();

    vector<double> areas;
    for (vector<cv::Point2f>& hull : hulls) {
        areas.push_back(cv::contourArea(hull));
    }

    auto p = sort_permutation(areas, [](double a, double b) {
        return a > b;
    });

    areas = apply_permutation(areas, p);
    clouds = apply_permutation(clouds, p);
    hulls = apply_permutation(hulls, p);
    objects = apply_permutation(objects, p);
    apply_permuation_matrix(adjacent, p);
    consolidated_indices = apply_permutation(consolidated_indices, p);

    for (size_t i = 0; i < hulls.size() - 1; ++i) {
        for (size_t j = i + 1; j < clouds.size(); ++j) {
            int num_inside_1 = 0;
            for (size_t k = 0; k < clouds[j].size(); ++k) {
                float point_dist = cv::pointPolygonTest(hulls[i], clouds[j][k], false);
                num_inside_1 += int(point_dist > 0.0f); // positive means inside
            }
            int num_inside_2 = 0;
            for (size_t k = 0; k < clouds[i].size(); ++k) {
                float point_dist = cv::pointPolygonTest(hulls[j], clouds[i][k], false);
                num_inside_2 += int(point_dist > 0.0f); // positive means inside
            }
            double fraction_inside_1 = double(num_inside_1)/double(clouds[j].size());
            double fraction_inside_2 = double(num_inside_2)/double(clouds[i].size());
            cout << "Fraction 1: " << fraction_inside_1 << endl;
            cout << "Fraction 2: " << fraction_inside_2 << endl;
            if (fraction_inside_1 > fraction_smaller &&
                fraction_inside_2 > fraction_larger && //) { // &&
                adjacent(i, j)) { // merge with larger object i
                cout << "Merging " << i << " and " << j << " with fraction: " << fraction_inside_1 << endl;
                merge_objects(objects[i], objects[j]);
                merge_adjacencies(adjacent, i, j);
                clouds[i].insert(clouds[i].end(), clouds[j].begin(), clouds[j].end());
                hulls[i] = compute_hull(clouds[i], hull_grow);
                //objects[i] = clone_merge_objects(objects[i], objects[j]);
                consolidated_indices[i].insert(consolidated_indices[i].end(), consolidated_indices[j].begin(), consolidated_indices[j].end());
                hulls.erase(hulls.begin() + j);
                clouds.erase(clouds.begin() + j);
                objects.erase(objects.begin() + j);
                consolidated_indices.erase(consolidated_indices.begin() + j);
                --j;
            }
        }
    }

    if (objects.size() < start_number_objects) {
        consolidate_objects(objects, hulls, clouds, adjacent, hull_grow, fraction_smaller, fraction_larger, consolidated_indices);
    }
}

Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> compute_adjacency(ObjectVec& objects)
{
    cout << "Computing adjacency..." << endl;

    float dist_threshold = 24.0;

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> adjacent(objects.size(), objects.size());
    adjacent.setZero();

    cout << "Computing contours..." << endl;

    vector<vector<vector<cv::Point> > > mask_contours(objects.size());
    for (size_t row = 0; row < objects.size(); ++row) {
        for (size_t i = 0; i < objects[row].masks.size(); ++i) {
            mask_contours[row].push_back(vector<cv::Point>{});
            vector<vector<cv::Point> > contours;
            cv::findContours(objects[row].masks[i], contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            for (const vector<cv::Point>& v : contours) {    
                for (const cv::Point& p : v) {
                    mask_contours[row].back().push_back(p);
                }
            }
            cout << "Contours length: " << mask_contours[row].back().size() << endl;
        }
    }

    cout << "Done computing contours..." << endl;

    for (size_t row = 0; row < objects.size(); ++row) {
        for (size_t col = 0; col < row; ++col) {
            // check if the masks of any frame are adjacent

            adjacent(row, col) = false;

            float minval = 10000.0;

            size_t i = 0; size_t j = 0;
            while (i < objects[row].frames.size() && j < objects[col].frames.size()) {
                if (objects[row].frames[i] < objects[col].frames[j]) {
                    ++i;
                    continue;
                }
                else if (objects[row].frames[i] > objects[col].frames[j]) {
                    ++j;
                    continue;
                }

                if (mask_contours[row][i].empty() || mask_contours[col][j].empty()) { // no object in either frame
                    ++i;
                    ++j;
                    continue;
                }

                // find minimum distance between masks
                for (const cv::Point& p : mask_contours[row][i]) {
                    auto iter = std::find_if(mask_contours[col][j].begin(), mask_contours[col][j].end(), [&](const cv::Point& q) {
                        return cv::norm(p-q) < dist_threshold;//float((p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y)) < dist_threshold*dist_threshold;
                    });
                    auto iter2 = std::min_element(mask_contours[col][j].begin(), mask_contours[col][j].end(), [&](const cv::Point& q1, const cv::Point& q2) {
                        return cv::norm(p-q1) < cv::norm(p-q2);//float((p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y)) < dist_threshold*dist_threshold;
                    });
                    if (minval > cv::norm(p-*iter2)) {
                        minval = cv::norm(p-*iter2);
                    }
                    if (iter != mask_contours[col][j].end()) {
                        adjacent(row, col) = true;
                        break;
                    }
                }

                if (adjacent(row, col)) {
                    break;
                }

                ++i;
                ++j;
            }

            cout << "Min distance " << " between object " << row << " and " << col << ": " << minval << endl;

            adjacent(col, row) = adjacent(row, col);
        }
    }

    cout << "Adjacency matrix: " << endl;
    cout << adjacent << endl;

    cout << "Done computing adjacency..." << endl;

    return adjacent;
}

// here we assume that they are already not overlapping
// we simply just find sort the objects based on size
// and check if any of the objects are within the convex
// hull in xy-plane or close to it among any of the smaller objects
vector<vector<size_t> > consolidate_objects(ObjectVec& objects, FrameVec& frames,
                                            const Eigen::Matrix4d& map_pose, float hull_grow,
                                            float fraction_smaller, float fraction_larger)
{
    vector<vector<cv::Point2f> > hulls;
    vector<vector<cv::Point2f> > clouds;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> adjacent = compute_adjacency(objects);

    vector<vector<size_t> > consolidated_indices;
    // ok, first convert all of the objects into matrix point clouds
    size_t counter = 0;
    for (SegmentedObject& obj : objects) {
        Eigen::Matrix<double, 3, Eigen::Dynamic> cloud = compute_object_cloud(obj, frames, map_pose);
        vector<cv::Point2f> points;
        for (size_t i = 0; i < cloud.cols(); ++i) {
            points.push_back(cv::Point2f(cloud(0, i), cloud(1, i)));
        }
        hulls.push_back(compute_hull(points, hull_grow));
        clouds.push_back(points);
        consolidated_indices.push_back(vector<size_t> {counter});
        ++counter;
    }

    consolidate_objects(objects, hulls, clouds, adjacent, hull_grow, fraction_smaller, fraction_larger, consolidated_indices);

    return consolidated_indices;
}
