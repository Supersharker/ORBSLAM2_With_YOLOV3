/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "pointcloudmapping.h"
#include "YOLOv3SE.h"
#include <Eigen/Geometry>
#include <KeyFrame.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <ctime>
#include <pcl/surface/gp3.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "Converter.h"
#include <boost/make_shared.hpp>

#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>


using namespace cv;
using namespace std;
using namespace pcl;
// using namespace APC;
using namespace pcl::io;
using namespace pcl::console;
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;


int pub_port = 6666;
#define NUM 5


int k = 500;
int min_size = 500;

PointCloudMapping::PointCloudMapping(double resolution_) {

    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    globalMap = boost::make_shared<PointCloud>();
    viewerThread = make_shared<thread>(bind(&PointCloudMapping::viewer, this));

}
void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

// 插入一个keyframe，会更新一次地图
void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{


    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;

    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );


    colorImgs.push_back( color.clone());

    depthImgs.push_back( depth.clone() );

    keyFrameUpdated.notify_one();
}




pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

//            if(p.b == p.g || p.b == p.r)
//                continue;
            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}






void PointCloudMapping::viewer()
{
    std::vector<cv::Scalar> colors;
    string line, number;
    ifstream f("color.txt");
    if (!f.good())
    {
        cout << "Cannot open file" << endl;
        exit(-1);
    }



    vector<int> v;
    while(std::getline(f, line))
    {
        istringstream is(line);

        while(std::getline(is, number, ','))
        {
            v.push_back(atoi(number.c_str()));
        }
        colors.push_back(cv::Scalar(v[0],v[1],v[2]));
        v.clear();
    }

    detector.Create("yolov3.weights", "yolov3.cfg", "coco.names");
    sleep(3);
    pcl::visualization::CloudViewer viewer("viewer");

    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }

        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }


        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }


        {

            for (size_t i=lastKeyframeSize; i<N ; i++) {

                cv::Mat tmp_color = colorImgs[i];
                char img[20];
                sprintf(img, "%s%d%s", "img/image", i, ".png");
                cv::imwrite(img,tmp_color);
                //PointCloud::Ptr pre_p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
                //PointCloud::Ptr p = regionGrowingSeg(pre_p);
                //*globalMap += *pre_p;
                //sleep(3);
                cv::Mat img_tmp_color = cv::imread(img);
                std::vector<BoxSE> boxes = detector.Detect(img_tmp_color, 0.5F);
                //continue;
                int n = boxes.size();
                for (int i = 0; i < n; i++) {
                   
                if(boxes[i].m_class_name == "keyboard" || boxes[i].m_class_name == "mouse" || boxes[i].m_class_name == "monitor" || boxes[i].m_class_name == "book" || boxes[i].m_class_name == "cup")
                {
                    cv::putText(img_tmp_color, detector.Names(boxes[i].m_class), boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0,colors[boxes[i].m_class], 2);
                    cv::rectangle(img_tmp_color, boxes[i].tl(), boxes[i].br(), colors[boxes[i].m_class], -1, 4);
                }

                    char img_laber[200];
                    sprintf(img_laber, "%s%d%s", "img_laber/image", i, ".png");
                    cv::imwrite(img_laber,img_tmp_color);
                }
                PointCloud::Ptr surf_p = generatePointCloud(keyframes[i], img_tmp_color, depthImgs[i]);
                //PointCloud::Ptr p = RegionGrowingSeg(surf_p);

                *globalMap += *surf_p;

            }


        }

        voxel.setInputCloud( globalMap );

        viewer.showCloud( globalMap );

        cout << "show global map, size=" << globalMap->points.size() << endl;
        lastKeyframeSize = N;

    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDWriter pcdwriter;
//write global point cloud map and save to a pcd file
    pcdwriter.write<pcl::PointXYZRGBA>("global_color.pcd", *globalMap);
    detector.Release();

}

void PointCloudMapping::obj2pcd(const std::string& inputFilename, const std::string& outputFilename)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Input stream
    std::ifstream is(inputFilename.c_str());

    // Read line by line
    for(std::string line; std::getline(is, line); )
    {
        std::istringstream in(line);

        std::string v;
        in >> v;
        if (v != "v") continue;

        // Read x y z
        float x, y, z;
        in >> x >> y >> z;
        cloud.push_back(pcl::PointXYZ(x, y, z));
    }

    is.close();

    // Save to pcd file
    pcl::io::savePCDFileBinaryCompressed(outputFilename, cloud);
}


void PointCloudMapping::PointXYZRGBAtoXYZ(const pcl::PointXYZRGBA& in,
                                pcl::PointXYZ& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
}

void PointCloudMapping::PointXYZLtoXYZ(const pcl::PointXYZL& in,
                                          pcl::PointXYZ& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;

}

void PointCloudMapping::PointCloudXYZRGBAtoXYZ(const pcl::PointCloud<pcl::PointXYZRGBA>& in,
                            pcl::PointCloud<pcl::PointXYZ>& out)
{
    out.width = in.width;
    out.height = in.height;
    for (size_t i = 0; i < in.points.size(); i++)
    {
        pcl::PointXYZ p;
        PointXYZRGBAtoXYZ(in.points[i],p);
        out.points.push_back (p);
    }
}



void PointCloudMapping::PointXYZRGBtoXYZRGBA(const pcl::PointXYZRGB& in,
                                pcl::PointXYZRGBA& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
    out.r = in.r; out.g = in.g; out.b = in.z; out.a =0;

}

void PointCloudMapping::PointCloudXYZRGBtoXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGB>& in,
                            pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
    out.width = in.width;
    out.height = in.height;
    for (size_t i = 0; i < in.points.size(); i++)
    {
        pcl::PointXYZRGBA p;
        PointXYZRGBtoXYZRGBA(in.points[i],p);
        out.points.push_back (p);
    }
}



bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
        return (true);
    else
        return (false);
}

bool
enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
        return (true);
    if (fabs (point_a_normal.dot (point_b_normal)) < 0.05)
        return (true);
    return (false);
}

bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (squared_distance < 10000)
    {
        if (fabs (point_a.intensity - point_b.intensity) < 8.0f)
            return (true);
        if (fabs (point_a_normal.dot (point_b_normal)) < 0.06)
            return (true);
    }
    else
    {
        if (fabs (point_a.intensity - point_b.intensity) < 3.0f)
            return (true);
    }
    return (false);
}
