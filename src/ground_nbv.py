#!/usr/bin/python3

import os
import rospy
import open3d as o3d
import numpy as np
from ctypes import * # convert float to uint32
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from project_mapping.srv import save_map, get_floor, get_nbv, align_floor
from project_mapping.msg import tuple_ndarray

import multiprocessing as mp
from multiprocessing import Pool
import copy as cp
import functions




class Map:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.pcds = []
        self.pcds_down = []
        self.combined_map = o3d.geometry.PointCloud()
        self.combined_map_down = o3d.geometry.PointCloud()
        self.map_2d = o3d.geometry.PointCloud()
        self.floor = o3d.geometry.VoxelGrid()
        self.floor_alignment = np.identity(4)
        self.up_to_date = False
        self.is_aligned = False
        self.next_best_views = [(np.asarray([0,0,0]) , np.asarray([0,0,0]))]
        

    def add_pc(self, pc):
        pc.estimate_normals()
        self.pcds.append(pc)
        self.up_to_date = False
        
    def add_pc_down(self, pc):
        pc.estimate_normals()
        self.pcds_down.append(pc)
        self.up_to_date = False
        
    def downsample_all(self):
        self.pcds_down.clear()
        for pc in self.pcds:
            self.pcds_down.append(pc.voxel_down_sample(voxel_size=self.voxel_size))
            
    def align_all(self):
        for i in range(len(self.pcds)):
            self.pcds[i].transform(self.floor_alignment)
            self.pcds_down[i].transform(self.floor_alignment)
            
        self.combined_map.transform(self.floor_alignment)
        self.combined_map_down.transform(self.floor_alignment)
        self.is_aligned = True
        
                    
    def combine_pc(self):
        pc_combined = o3d.geometry.PointCloud()
        pc_down_combined = o3d.geometry.PointCloud()
        for pc, pc_down in zip(self.pcds, self.pcds_down):
            pc_combined += pc
            pc_down_combined += pc_down
        
        self.pcds = [pc_combined]
        self.pcds_down = [pc_down_combined]
        self.combined_map = pc_combined
        self.combined_map_down = pc_down_combined
        self.up_to_date = True
        
    def save(self, path, downsampled=False):
        if not self.up_to_date:
            print("Map is not the latest version")
        if downsampled:
            o3d.io.write_point_cloud(path, self.combined_map_down)
        else:
            o3d.io.write_point_cloud(path, self.combined_map)
            
    def create_2d(self):
        cl, ind = self.combined_map.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.8)

        pcd = cp.deepcopy(self.combined_map.select_by_index(ind))
        pcd = pcd.voxel_down_sample(voxel_size=0.1)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(40)
        self.map_2d = functions.filter_by_normal(pcd)
        
        
        


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
def register(map):
    voxel_size = map.voxel_size
    
    
    print("Full registration ...")

    # calculate correspondance distances based on voxel size
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    # calculate pose graph for downsamples pointclouds; verbosity level can be set: Debug, Error, Info, Warning
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(map.pcds_down,
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

    # create empty pointcloud and combine input pointclouds with corresponding transform
    # original pointclouds are combined here; change from pcds to pcds_down if downsampled pointcloud is desired
    print("Transform points and display")
    
    for point_id in range(len(map.pcds)):
        print(pose_graph.nodes[point_id].pose)
        map.pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        map.pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        


def registration(data, map):
    voxel_size = map.voxel_size
    
    
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    
    
    print(len(map.pcds))
    pcd = convertCloudFromRosToOpen3d(data)
    pcd.transform(map.floor_alignment)
    pcd.estimate_normals()
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    if (len(map.pcds) == 0):
        map.add_pc(pcd)
        map.add_pc_down(pcd_down)
    else:
        
        transformation_icp, information_icp = pairwise_registration(pcd_down, map.pcds_down[-1], max_correspondence_distance_coarse, max_correspondence_distance_fine)
        pcd.transform(transformation_icp)
        pcd_down.transform(transformation_icp)
        
        map.add_pc(pcd)
        map.add_pc_down(pcd_down)
        
        map.up_to_date = False
        o3d.visualization.draw_geometries(map.pcds)
    
    if ((len(map.pcds_down) % 5) == 0):
        register(map)
        map.combine_pc()
    

        
    
    rospy.loginfo("Received a PointCloud2 message")

def save_map_handler(req, map):
    
    if (len(map.pcds) == 0):
        return "can not save. No pointcloud data available."
    
    filename = req.filename + ".pcd"
    path = os.getcwd() + "/" + filename
    register(map)
    
    map.combine_pc()
    map.save(path)
    
    return path



def get_floor_handler(req, map):


    
    if not map.up_to_date:
        register(map)
        map.combine_pc()
        
    
        
    map.create_2d()
    
    oboxes = map.map_2d.detect_planar_patches(
    normal_variance_threshold_deg=50,
    coplanarity_deg=85,
    outlier_ratio=0.75,
    min_plane_edge_length=2,
    min_num_points=10,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))

    #print("Detected {} patches".format(len(oboxes)))

    meshes = []
    for obox in oboxes:

        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        
        meshes.append(mesh)

    hor_patches, ver_patches = functions.divide_meshes_hor_ver(meshes)
    corr, corr_tuples = functions.group_corresponding_meshes([],[],hor_patches, "horizontal")

    corr, corr_tuples = functions.group_corresponding_meshes(corr, corr_tuples, ver_patches, "vertical")
    midpoints, marker_meshes = functions.find_midpoints(corr_tuples)
    
    hull, _ = map.map_2d.compute_convex_hull()

    uniform_pc = functions.hull_to_uniform_pc(hull, 0.2, [1,0,0])
    pcd_flat = functions.create_uniform_pc(map.map_2d, 0.1, [0,0,0])

    

    uniform_pc.paint_uniform_color([1,0,0])
    pcd_flat.paint_uniform_color([0,0,0])
    
    void_area, known_points = functions.extract_void_area(uniform_pc, pcd_flat, midpoints)
    valid_area = uniform_pc.select_by_index(list(known_points))
    valid_area_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(valid_area.paint_uniform_color([0,1,0]), 0.2)
    map.floor = valid_area_voxel
    o3d.visualization.draw_geometries([map.floor, map.map_2d])
    map.up_to_date = True
    return []



def get_nbv_handler(req, map):
    if not map.up_to_date:
        register(map)
        map.combine_pc()
        map.create_2d()
    get_floor_handler(None, map)
    
    if len(map.floor.get_voxels()) < 100:
        return "no floor space detected. can't start nbv planning"
    
    print("1st")
    pcd = cp.deepcopy(map.combined_map)
    pcd.estimate_normals()
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=0.8)

    pcd = pcd.select_by_index(ind)
    pcd = pcd.voxel_down_sample(voxel_size=map.voxel_size)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(40)
    boundaries = functions.detect_boundaries(pcd)
    geometries = functions.detect_boundary_patches(boundaries)
    
    print("start ran")
    ranked_views = functions.rank_all_views(geometries, (2,10), 10 )
    ranked_views = functions.sort_xd_list(ranked_views, 0, "descending")
    hit_areas = [view[0] for view in ranked_views]
    best_positions = [view[1] for view in ranked_views]
    best_orientations = [view[2] for view in ranked_views]
    
    
    if not map.is_aligned:
        align_floor_handler(None, map)
        
        
    best_views = functions.find_best_valid_view(best_positions, best_orientations, map.floor)
        
    
    map.next_best_views = best_views
    
    map.up_to_date = True
    
    print(map.next_best_views[0][0])
    print(map.next_best_views[0][1])
    
    views = []
    first = True
    scale = 1
    for view in map.next_best_views:
        if first:
            views.append(functions.create_arrow_pos_ori(view[0], view[1], scale=scale, color=[0,1,0]))
            first = False
            scale *= 0.8
        else:
            views.append(functions.create_arrow_pos_ori(view[0], view[1], scale=scale, color=[1,0,0]))
            scale *= 0.8
    
    arrow1 = functions.create_arrow_pos_ori([0,0,0], [1,0,0])
    arrow2 = functions.create_arrow_pos_ori([0,0,0], [0,1,0], color = [0,1,0])
    arrow3 = functions.create_arrow_pos_ori([0,0,0], [0,0,1], color = [0,0,1])
    
    
    
    o3d.visualization.draw_geometries([map.combined_map] + views + geometries + [arrow1 , arrow2 , arrow3])
    
    
    response_msg = tuple_ndarray()
    response_msg.vector1.data = map.next_best_views[0][0]
    response_msg.vector2.data = map.next_best_views[0][1]
    
    
    
    return response_msg


def align_floor_handler(req, map):
    ground_plane = np.asarray([0,0,1,0])
    pcd_ground_plane, inliers = functions.detect_ground_plane(map.combined_map)
    trans = functions.calculate_floor_alignment_matrix(pcd_ground_plane)
    inlier_cloud = map.combined_map.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    
    o3d.visualization.draw_geometries([map.combined_map, inlier_cloud])
    
    
    map.floor_alignment = trans
    map.align_all()
    
    
    
    
    
    return []


def main():
    rospy.init_node("pointcloud_icp", anonymous=True)
    service_name_save = rospy.get_name() + "/save_map"
    service_name_floor = rospy.get_name() + "/get_floor"
    service_name_nbv = rospy.get_name() + "/get_nbv"
    service_name_align = rospy.get_name() + "/algin_floor"
    map = Map(voxel_size=0.1)
   
    save_map_handler_lambda = lambda x: save_map_handler(x,map)
    get_floor_handler_lambda = lambda x: get_floor_handler(x,map)
    get_nbv_handler_lambda = lambda x: get_nbv_handler(x,map)
    align_floor_handler_lambda = lambda x: align_floor_handler(x,map)
    #s1 = rospy.Service(service_name_save, save_map, save_map_handler_lambda)
    #s2 = rospy.Service(service_name_floor, get_floor, get_floor_handler_lambda)
    s1 = rospy.Service("/save_map", save_map, save_map_handler_lambda)
    s2 = rospy.Service("/get_floor", get_floor, get_floor_handler_lambda)
    s3 = rospy.Service("/get_nbv", get_nbv, get_nbv_handler_lambda)
    s4 = rospy.Service("/algin_floor", align_floor, align_floor_handler_lambda)
        
    rospy.Subscriber("/periodic_snapshotter/assembled_cloud_2", PointCloud2, registration, map)
    #rospy.Subscriber("/GETjag/laser_cloud_last", PointCloud2, registration, map)
    
    rospy.spin()

    

if __name__ == '__main__':
    main()
 
