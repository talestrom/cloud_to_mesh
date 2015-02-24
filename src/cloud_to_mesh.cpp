#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include <visualization_msgs/Marker.h>
#include <sstream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/conversions.h>
#include <pcl/io/vtk_io.h>
#include <pcl/common/common.h>
#include <pcl/surface/mls.h>
#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/features/normal_3d_omp.h>
#include <stdio.h>
#include <time.h>
#include <tf/transform_listener.h>
#include <cmath>

#include "tf/message_filter.h"
#include "message_filters/subscriber.h"

#include <pcl/surface/ear_clipping.h>
#include <pcl/surface/poisson.h>

class PointCloudToMesh {
private:
    ros::NodeHandle nh;
    ros::NodeHandle pnh;
    ros::Subscriber sub;
    ros::Publisher pub;
    ros::Publisher pub_marker;
    ros::Publisher pub_filtered;

    std::string sensor_frame_id;
    std::string method;

    //filtering and smoothing
    double filter_leafsize;
    bool smoothing;

    // variables for greedy projection
    double greedy_radius;
    double greedy_mu;
    int greedy_neighbors;
    double greedy_surface_angle;

    //variables for convex hull
    double edge_angle;
    double max_area;
    double normal_angle;

    //variables for Poisson
    double max_distance;

    //tf
    tf::TransformListener listener;
    tf::StampedTransform transform;
    tf::MessageFilter<geometry_msgs::PointStamped>* tf_filter;

    void unpack_rgb(pcl::PointXYZRGB& point, std_msgs::ColorRGBA& barva)
    {
        // unpack rgb into r/g/b
        uint32_t rgb = *reinterpret_cast<uint32_t*>(&point.rgb);
        uint8_t r = (rgb >> 16) & 0x0000ff;
        uint8_t g = (rgb >> 8)  & 0x0000ff;
        uint8_t b = (rgb)       & 0x0000ff;
        // unpack rgb into r/g/b
        barva.r = float(r)/255;
        barva.g = float(g)/255;
        barva.b = float(b)/255;
        barva.a = 1;
    }

    void normals_trees(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                       pcl::PointCloud<pcl::PointNormal>::Ptr& cloud_with_normals,
                       pcl::search::KdTree<pcl::PointNormal>::Ptr& tree2)
    {
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (cloud);
        n.setInputCloud (cloud);
        n.setSearchMethod (tree);
        n.setKSearch (20);
        n.compute (*normals);
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
        tree2->setInputCloud(cloud_with_normals);
    }

    void greedy_triangles(pcl::PolygonMesh& triangles, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                          double radius, double mu, int neighbors, double surface_angle)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
        normals_trees(cloud, cloud_with_normals, tree2);

        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
        gp3.setSearchRadius (radius);
        gp3.setMu (mu);
        gp3.setMaximumNearestNeighbors (neighbors);
        gp3.setMaximumSurfaceAngle(surface_angle); // 45 degrees
        gp3.setMinimumAngle(M_PI/23); // 10 degrees
        gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
        gp3.setNormalConsistency(false);
        gp3.setInputCloud (cloud_with_normals);
        gp3.setSearchMethod (tree2);
        gp3.reconstruct (triangles);
    }

    void poisson_triangles(pcl::PolygonMesh& triangles, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_xyz)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
        normals_trees(cloud_xyz, cloud_with_normals, tree2);//*/

        pcl::Poisson<pcl::PointNormal> poisson;
        poisson.setDepth(6);
        poisson.setInputCloud(cloud_with_normals);
        poisson.setScale(1.01);
        poisson.setSamplesPerNode(5.0);
        poisson.setManifold(false);
        poisson.reconstruct(triangles);

        //pcl::io::saveVTKFile("mesh.vtk", triangles);
    }

    void transform_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_in,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_out,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_moved,
                         tf::StampedTransform transform)
    {
        cloud_out->points.resize(cloud_in->size());
        cloud_out->width = cloud_in->width;
        cloud_out->height = cloud_in->height;
        cloud_out->header = cloud_in->header;

        cloud_moved->points.resize(cloud_in->size());
        cloud_moved->width = cloud_in->width;
        cloud_moved->height = cloud_in->height;
        cloud_moved->header = cloud_in->header;

        double x_t = transform.getOrigin().x();
        double y_t = transform.getOrigin().y();
        double z_t = transform.getOrigin().z();

        for (size_t i = 0; i < cloud_in->points.size(); i++)
        {
          double x = cloud_in->points[i].x;
          double y = cloud_in->points[i].y;
          double z = cloud_in->points[i].z;

          double length = sqrt(pow((x - x_t),2) + pow((y - y_t),2) + pow((z - z_t),2));

          cloud_out->points[i].x = (x - x_t)/length ;
          cloud_out->points[i].y = (y - y_t)/length ;
          cloud_out->points[i].z = (z - z_t)/length ;

          cloud_out->points[i].r = cloud_in->points[i].r;
          cloud_out->points[i].g = cloud_in->points[i].g;
          cloud_out->points[i].b = cloud_in->points[i].b;

          cloud_moved->points[i].x = (x - x_t);
          cloud_moved->points[i].y = (y - y_t);
          cloud_moved->points[i].z = (z - z_t);

          cloud_moved->points[i].r = cloud_in->points[i].r;
          cloud_moved->points[i].g = cloud_in->points[i].g;
          cloud_moved->points[i].b = cloud_in->points[i].b;
        }
    }

    void init_Marker(visualization_msgs::Marker& m, const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        m.header = msg->header;
        m.header.stamp = ros::Time::now();
        m.ns = "cloud_to_mesh";
        m.id = 0;
        m.type = visualization_msgs::Marker::TRIANGLE_LIST;
        m.action = visualization_msgs::Marker::ADD;
        m.pose.position.x = 0;
        m.pose.position.y = 0;
        m.pose.position.z = 0;
        m.pose.orientation.x = 0.0;
        m.pose.orientation.y = 0.0;
        m.pose.orientation.z = 0.0;
        m.pose.orientation.w = 1.0;
        m.scale.x = 1.0;
        m.scale.y = 1.0;
        m.scale.z = 1.0;
        m.lifetime = ros::Duration();
    }

    void input_points_greedy(visualization_msgs::Marker& m, pcl::PolygonMesh& triangles,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
    {
        for (int i=0; i < triangles.polygons.size() ; i++)
         {
            for(int j = 0; j < triangles.polygons[i].vertices.size(); j++)
            {
                   pcl::PointXYZRGB point = cloud->points[triangles.polygons[i].vertices[j]];
                   geometry_msgs::Point bod;

                   bod.x = cloud->points[triangles.polygons[i].vertices[j]].x;
                   bod.y = cloud->points[triangles.polygons[i].vertices[j]].y;
                   bod.z = cloud->points[triangles.polygons[i].vertices[j]].z;
                   m.points.push_back(bod);

                   std_msgs::ColorRGBA barva;
                   unpack_rgb(point,barva);
                   m.colors.push_back(barva);
            }
         }
    }

    void input_points_poisson(visualization_msgs::Marker& m, pcl::PolygonMesh& triangles,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_col, double max_distance)
    {
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud (cloud_col);
        std::vector<int> map;
        std::vector<double> distances;

        for (int k=0; k < cloud->points.size(); k++)
        {
            std::vector<int> indice;
            std::vector<float> distance;
            pcl::PointXYZRGB point;
            point.x = cloud->points[k].x;
            point.y = cloud->points[k].y;
            point.z = cloud->points[k].z;
            tree->nearestKSearch(point, 1, indice , distance);
            map.push_back(indice[0]);
            distances.push_back(distance[0]);
        }

        for (int i=0; i < triangles.polygons.size() ; i++)
         {
            int ak = triangles.polygons[i].vertices[0];
            int bk = triangles.polygons[i].vertices[1];
            int ck = triangles.polygons[i].vertices[2];

            if ((distances[ak] < max_distance) && (distances[bk] < max_distance) &&
                    (distances[ck] < max_distance))
            {
              for(int j = 0; j < triangles.polygons[i].vertices.size(); j++)
                {
                   int k = triangles.polygons[i].vertices[j];
                   pcl::PointXYZ point = cloud->points[k];
                   geometry_msgs::Point bod;
                   bod.x = point.x;
                   bod.y = point.y;
                   bod.z = point.z;
                   m.points.push_back(bod);

                   pcl::PointXYZRGB close_point = cloud_col->points[map[k]];
                   std_msgs::ColorRGBA barva;
                   unpack_rgb(close_point,barva);
                   m.colors.push_back(barva);
                }
            }
         }
    }

    void map_construction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr& hull, std::vector<int>& map)
    {
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud (cloud);

        for (int k=0; k < hull->points.size(); k++)
        {
            std::vector<int> indice;
            std::vector<float> distance;
            pcl::PointXYZRGB point = hull->points[k];
            tree->nearestKSearch(point, 1, indice , distance);
            map.push_back(indice[0]);
        }
    }

    void input_points_convexhull(visualization_msgs::Marker& m, std::vector<pcl::Vertices>& polygons,
                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr& hull,
                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_transformed,
                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_moved)
    {
        double min_edge_angle_cos = std::cos(M_PI * edge_angle / 180.0);
        double min_normal_angle_cos = std::cos(M_PI * normal_angle / 180.0);

        std::vector<int> map;
        map_construction(cloud_transformed, hull, map);

        for (int i=0; i < polygons.size() ; i++)
         {
            int ak = polygons[i].vertices[0];
            int bk = polygons[i].vertices[1];
            int ck = polygons[i].vertices[2];

            pcl::PointXYZRGB a = hull->points[ak];
            pcl::PointXYZRGB b = hull->points[bk];
            pcl::PointXYZRGB c = hull->points[ck];
            Eigen::Vector3f a_vec = a.getVector3fMap();
            Eigen::Vector3f b_vec = b.getVector3fMap();
            Eigen::Vector3f c_vec = c.getVector3fMap();

            double a_b = a_vec.dot(b_vec); //cos of the angle for an adge AB
            double c_b = c_vec.dot(b_vec); // BC
            double a_c = a_vec.dot(c_vec); // AC

            pcl::PointXYZRGB A = cloud_moved->points[map[ak]];
            pcl::PointXYZRGB B = cloud_moved->points[map[bk]];
            pcl::PointXYZRGB C = cloud_moved->points[map[ck]];
            Eigen::Vector3f A_vec = A.getVector3fMap();
            Eigen::Vector3f B_vec = B.getVector3fMap();
            Eigen::Vector3f C_vec = C.getVector3fMap();
            Eigen::Vector3f centroid = (A_vec + B_vec + C_vec) / 3;

            Eigen::Vector3f u = B_vec - A_vec;
            Eigen::Vector3f v = C_vec - A_vec;

            Eigen::Vector3f norm = v.cross(u);
            double double_area = norm.norm();

            double view_normal_cos = std::abs(centroid.dot(norm)/(double_area*centroid.norm()));

            if ((a_b > min_edge_angle_cos) && (c_b > min_edge_angle_cos) && (a_c > min_edge_angle_cos)
                    && (view_normal_cos > min_normal_angle_cos))
            {
              //ROS_INFO("kosinus uhlu (> 0.342?) %f", view_normal_cos);
              for (int j = 0; j < polygons[i].vertices.size(); j++)
                 {
                   int k = polygons[i].vertices[j];

                   geometry_msgs::Point bod;
                   bod.x = cloud->points[map[k]].x;
                   bod.y = cloud->points[map[k]].y;
                   bod.z = cloud->points[map[k]].z;
                   m.points.push_back(bod);

                   std_msgs::ColorRGBA barva;
                   unpack_rgb(cloud->points[map[k]],barva);
                   m.colors.push_back(barva);
                 }
            }
         }
    }


    void init_Mls(pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB>& mls,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
    {
        mls.setInputCloud(cloud);
        mls.setSearchRadius(0.5);
        mls.setPolynomialFit(true);
        mls.setPolynomialOrder(2);
        mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB>::SAMPLE_LOCAL_PLANE);
        mls.setUpsamplingRadius(0.03);
        mls.setUpsamplingStepSize(0.03);
    }

    void xyzrgb_to_xyz( pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_xyzrgb,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_xyz)
    {
        cloud_xyz->points.resize(cloud_xyzrgb->size());
        cloud_xyz->width = cloud_xyzrgb->width;
        cloud_xyz->height = cloud_xyzrgb->height;
        cloud_xyz->header = cloud_xyzrgb->header;
        for (size_t i = 0; i < cloud_xyzrgb->points.size(); i++)
        {
            cloud_xyz->points[i].x = cloud_xyzrgb->points[i].x;
            cloud_xyz->points[i].y = cloud_xyzrgb->points[i].y;
            cloud_xyz->points[i].z = cloud_xyzrgb->points[i].z;
        }
    }
    void voxel_grid_filter(const sensor_msgs::PointCloud2 msg,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_filtered,
                           double leafsize)
    {
        pcl::PCLPointCloud2::Ptr pcl_pc (new pcl::PCLPointCloud2());
        pcl_conversions::toPCL(msg,*pcl_pc);
        pcl::VoxelGrid<pcl::PCLPointCloud2 > sor;
        sor.setInputCloud(pcl_pc);
        sor.setLeafSize(leafsize, leafsize, leafsize);
        pcl::PCLPointCloud2::Ptr cloud_filtered_2 (new pcl::PCLPointCloud2());
        sor.filter(*cloud_filtered_2);
        pcl::fromPCLPointCloud2(*cloud_filtered_2,*cloud_filtered);
    }

    void chatterCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        pnh.param("sensor_frame_id", sensor_frame_id, sensor_frame_id);
        pnh.param("method", method, method);
        pnh.param("filter_leafsize", filter_leafsize, filter_leafsize);
        pnh.param("edge_angle", edge_angle, edge_angle);
        pnh.param("max_area", max_area, max_area);
        pnh.param("normal_angle", normal_angle, normal_angle);

        pnh.param("poisson_distance", max_distance, max_distance);
        smoothing = false;

        // Timers
        ros::Time begin; ros::Time end; ros::Duration duration;

        // Cloud initialization
        pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *cloud);        
        pcl::PointCloud< pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);

        // FILTERING 0.14 seconds
        pcl::PointCloud< pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
        //cloud_filtered = cloud;
        voxel_grid_filter(*msg, cloud_filtered, filter_leafsize);

        // getting rid of the color
        xyzrgb_to_xyz(cloud_filtered, cloud_xyz);

        // PolygonMesh
        pcl::PolygonMesh triangles;      

        // Marker initialization
        visualization_msgs::Marker m;
        init_Marker(m, msg);

        // MOVING LEAST SQUARES takes 0.1 (if the sample is noisy)
        if (smoothing)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_smoothed (new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
            init_Mls(mls, cloud_filtered);
            mls.process(*cloud_smoothed);
            xyzrgb_to_xyz(cloud_smoothed, cloud_xyz);
            cloud_filtered = cloud_smoothed;
        }

        // MESHING WITHOUT POINT MODIFICATION
        // GREEDY PROJECTION  0.15 with filtered cloud
        begin = ros::Time::now();
        if (method == "greedy_projection")
        {
            greedy_radius = 0.5;
            greedy_mu = 2.5;
            greedy_neighbors = 100;
            greedy_surface_angle = M_PI/4;
            greedy_triangles(triangles, cloud_xyz, greedy_radius, greedy_mu, greedy_neighbors, greedy_surface_angle);
            input_points_greedy(m, triangles, cloud_filtered);
        }
        end = ros::Time::now();
        duration = end - begin;
        ROS_INFO("RYCHLOST JE %f", duration.toSec());

        //CONVEXHULL takes 0.15 seconds with previous filtering (0.7 with smoothing)
        if (method == "convexhull_projection")
        {
            bool if_exists = listener.waitForTransform(msg->header.frame_id,
                            sensor_frame_id, msg->header.stamp, ros::Duration(2));
            if(if_exists)
            {
                listener.lookupTransform(msg->header.frame_id, sensor_frame_id, msg->header.stamp, transform);
            }
            pcl::PointCloud< pcl::PointXYZRGB>::Ptr hull (new pcl::PointCloud<pcl::PointXYZRGB>);
            std::vector<pcl::Vertices> polygons;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_moved (new pcl::PointCloud<pcl::PointXYZRGB>);
            transform_cloud(cloud_filtered, cloud_transformed, cloud_moved, transform);

            // takes 0.05 seconds
            pcl::ConvexHull<pcl::PointXYZRGB> convex_hull;
            convex_hull.setInputCloud(cloud_transformed);
            convex_hull.reconstruct(*hull, polygons);

            // takes 0.098 - 0.143 seconds
            input_points_convexhull(m, polygons,hull, cloud_filtered, cloud_transformed, cloud_moved);
        }

        //MESHING WITH POINT MODIFICATION
        //POISSON PROJECTION takes 0.73 seconds (0.93 with smoothing)
        if (method == "poisson_projection")
        {
            poisson_triangles(triangles, cloud_xyz);
            pcl_conversions::toPCL(msg->header, triangles.cloud.header);
            pcl::fromPCLPointCloud2(triangles.cloud, *cloud_xyz);
            input_points_poisson(m, triangles, cloud_xyz, cloud_filtered, max_distance);
        }

        // PUBLISHING
        pub_marker.publish(m);
        pub.publish(cloud_filtered);
    }

public:
    PointCloudToMesh(): nh(), pnh("~"), sensor_frame_id("/laser") {
        pnh.param("sensor_frame_id", sensor_frame_id, sensor_frame_id);
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/cloud", 1);
        pub_filtered = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/cloud_filtered",1);
        pub_marker = nh.advertise<visualization_msgs::Marker>("/mesh_marker",1);
        sub = nh.subscribe<sensor_msgs::PointCloud2>("/dynamic_point_cloud", 1000, &PointCloudToMesh::chatterCallback, this);
    }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cloud_to_mesh");
  PointCloudToMesh node;
  ros::spin();
  ROS_INFO("SDKJHSDK");
  return 0;
}
