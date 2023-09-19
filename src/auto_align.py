#!/usr/bin/python3

import rospy
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from project_mapping.srv import save_map


pcds = []
pcds_down = []
aligned_pc = o3d.geometry.PointCloud()
voxel_size = 0.1

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)


def convertCloudFromRosToOpen3d(ros_cloud):
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

    # Check empty
    open3d_cloud = o3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None
        
    # Set open3d_cloud
    if "rgb" in field_names:
        cloud_data = [cloud[0:4] for cloud in cloud_data]
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

        # combine
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        cloud_data = [cloud[0:3] for cloud in cloud_data]
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

    # return
    return open3d_cloud



def downsample_pointcloud(pointcloud, voxel_size=0.1):
    
    pointcloud.estimate_normals()
    
    # downsample original pointcloud
    pcd_down = pointcloud.voxel_down_sample(voxel_size=voxel_size)

    return pcd_down



# pairwise registration of 2 pointclouds; run 2 icp passes, one for coarse, one for fine matching, return transformation and information matrix
def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # final calculated transformation
    transformation_icp = icp_fine.transformation

    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


# full registration with list of pointclouds; returns pose graph containing pointclouds and corresponding transformations
def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    # create pose graph and append first node; first node is base frame
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    # go through list of pointclouds and do pairwise registration
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)

            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case, if pointclouds ar direct neighbours
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                 target_id,
                                                                                 transformation_icp,
                                                                                 information_icp,
                                                                                 uncertain=False))
            else:  # loop closure case, if pointclouds are not direct neighbours
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                                 target_id,
                                                                                 transformation_icp,
                                                                                 information_icp,
                                                                                 uncertain=True))
    return pose_graph

# function to start the procedure of ICP registration; input pointclouds, output filename and voxel size are given
def register(pcds, pcds_down, output, voxel_size):
    print("Full registration ...")

    # calculate correspondance distances based on voxel size
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    # calculate pose graph for downsamples pointclouds; verbosity level can be set: Debug, Error, Info, Warning
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    # set options for pose graph optimization
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)

    # optimize pose graph with given parameters; verbosity level can be set: Debug, Error, Info, Warning
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(pose_graph,
                                                       o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                       o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                       option)

    # create empty pointcloud and combine input pointclouds with correstponding transform
    # original pointclouds are combined here; change from pcds to pcds_down if downsampled pointcloud is desired
    print("Transform points and display")
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        print(pose_graph.nodes[point_id].pose)
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    # save output as .pcd file and visualize
    #o3d.io.write_point_cloud(output, pcd_combined)
    o3d.visualization.draw_geometries([pcd_combined])


def callback(data):
    
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    
    
    pcd = convertCloudFromRosToOpen3d(data)
    pcd.estimate_normals()
    
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    if (len(pcds_down) == 0):
        pcds.append(pcd)
        pcds_down.append(pcd_down)
    else:
        
        transformation_icp, information_icp = pairwise_registration(pcd_down, pcds_down[-1], max_correspondence_distance_coarse, max_correspondence_distance_fine)
        pcd.transform(transformation_icp)
        pcd_down.transform(transformation_icp)
        pcds.append(pcd)
        pcds_down.append(pcd_down)
        o3d.visualization.draw_geometries(pcds)
    
    rospy.loginfo("Received a PointCloud2 message")

def save_map_handler(req):
    print(req.input_string)
    #register(pcds, pcds_down, "req.input_string", voxel_size)
    return "hello"

def main():
    

    #register(pcds, pcds_down, output_pc, voxel_size)

    rospy.init_node('pointcloud_subscriber', anonymous=True)
    s = rospy.Service('save_map', save_map, save_map_handler)
        
    rospy.Subscriber('/periodic_snapshotter/assembled_cloud_2', PointCloud2, callback)
    rospy.spin()

    

if __name__ == '__main__':
    main()
 
