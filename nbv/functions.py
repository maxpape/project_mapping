import open3d as o3d
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import copy as cp
#import open3d.core as o3c
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
import time
from scipy.spatial.transform import Rotation
from iteration_utilities import deepflatten
from mpl_toolkits.mplot3d import Axes3D

def create_box_at_point(point, size=(0.1,0.1,0.1), color=[0,1,0]):
    # function to create marker for visualization
    size_x, size_y, size_z = size
    box = o3d.geometry.TriangleMesh.create_box(size_x, size_y, size_z)
    box.paint_uniform_color(color)
    
    box.translate(point, False)
    
    
    return box

def filter_by_normal(pcd, hight_variance = 0.01):
    
    #check if Pointcloud has normals; calculate normals if not
    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(40)
    
    ind = []
    # go through all points: if normals dominantly point towards x or y direction and not towards z direction:
    # set random hight in range 0 to hight_variance
    # select only the points with altered hight
    for i in range(len(pcd.points)):
        if ((np.abs(pcd.normals[i][2]) < 0.3) and ((np.abs(pcd.normals[i][0]) > 0.8) or (np.abs(pcd.normals[i][1]) > 0.8))):
            pcd.points[i][2] = np.random.rand()*hight_variance
        else:
            ind.append(i)
    filtered_pcd = pcd.select_by_index(ind, invert=True)
    return filtered_pcd


def are_vectors_perpendicular(v1, v2, threshold):
    
    # Normalize the vectors to unit length
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle = np.degrees(angle)
    
    # direction does not matter, reduce angle range from 0 to 180 degrees
    if (180 <= angle < 360):
        angle -= 180

    # vectors are perpendicular, if angle is between 90° +- threshold    
    if ((90-threshold) <= angle <= (90+threshold)):
        return True
    else:
        return False

def find_midpoint_between_planes(plane1, plane2):
    # calculate midpoint between two planes

    center1 = plane1.get_center()
    center2 = plane2.get_center()

    vec_1_2 = center2 - center1

    midpoint = center1 + vec_1_2 / 2
    return midpoint

def find_midpoints(corr_tuples):
    # find midpoints for a list of corresponding meshes (walls)
    # return midpoints as well as markers for visualization
    midpoints = []
    marker_meshes = []

    for tup in corr_tuples:
        midpoint = find_midpoint_between_planes(tup[0], tup[1])
    
    
        midpoints.append(midpoint)
        marker_meshes.append(create_box_at_point(midpoint))
    return midpoints, marker_meshes


def devide_meshes_hor_ver(meshes):
    
    # take first mesh, check if its perpendicular to vector [1,0,0]:
    # yes: its a vertical patch
    # no: its a horizontal patch
    # tmp is used to sort following patches according to first classification of vertical/horizontal
    first = meshes[0]
    tmp = False
    ver_patches = []
    hor_patches = []
    if (are_vectors_perpendicular(first.vertex_normals[0], np.asarray([1,0,0]), 15)):
        ver_patches.append(first)
    else:
        hor_patches.append(first)
        tmp = True
    

    for i in range(1,len(meshes)):
        patch_normal = meshes[i].vertex_normals[0]

        if (are_vectors_perpendicular(first.vertex_normals[0], patch_normal, 10)):
            if (tmp):
                ver_patches.append(meshes[i])
            else:
                hor_patches.append(meshes[i])
        else:
            if (tmp):
                hor_patches.append(meshes[i])
            else:
                ver_patches.append(meshes[i])



    return hor_patches, ver_patches


def get_mesh_distance(mesh1, mesh2, orientation):
    # define wether horizontal or vertical direction is needed
    o = {
    "vertical": 0,
    "horizontal": 1}

    # choose center value: either x or y coordinate according to chosen direction
    bb1 = mesh1.get_oriented_bounding_box()
    bb1_center = bb1.get_center()[1-o[orientation]]
    bb2 = mesh2.get_oriented_bounding_box()
    bb2_center = bb2.get_center()[1-o[orientation]]
    
    # coordinates can have + or - sign. take into account for distance calculation
    if (bb1_center < 0 < bb2_center):
        dist = -bb1_center + bb2_center
    elif (bb2_center < 0 < bb1_center):
        dist = bb1_center - bb2_center
    else:
        dist = np.abs(bb1_center-bb2_center)
        
    return dist


def mesh_correspondance(mesh1, mesh2, orientation):
    # function to check wether two parallel meshes are "corresponding" to each other
    # => both meshes probably belong to opposing walls
    o = {
    "vertical": 0,
    "horizontal": 1}
    
    # get bounds, get centers
    bb1 = mesh1.get_oriented_bounding_box()
    bb2 = mesh2.get_oriented_bounding_box()
    bb1_min = bb1.get_min_bound()
    bb1_max = bb1.get_max_bound()
    bb2_min = bb2.get_min_bound()
    bb2_max = bb2.get_max_bound()

    # differentiate between center in horizontal and vertical direction
    bb1_center = bb1.get_center()[o[orientation]]
    bb2_center = bb2.get_center()[o[orientation]]
    
    bb1_center2 = bb1.get_center()[1-o[orientation]]
    bb2_center2 = bb2.get_center()[1-o[orientation]]
    
    # calculate distance between parralel meshes
    # if distance is to small, they are probably not walls of one room
    # more likely the belong to inner space of wall
    dist = get_mesh_distance(mesh1, mesh2, orientation)
    if (dist < 1):
        return False
    

    # center of at least one mesh (wall) must lie in between edge points of other wall
    # |______|      this example is valid (center left wall between edge points of right wall)
    # |      |      walls "correspond"
    #        |
    #
    # |_______      this one is not valid (center of both walls do not lie in between edge points of other wall)
    # |      |
    # _______|
    #        |
    #        |
    if (bb1_min[o[orientation]] < bb2_center < bb1_max[o[orientation]]):
        return True
    if (bb2_min[o[orientation]] < bb1_center < bb2_max[o[orientation]]):

        return True
    
    
    return False



def find_nearest_mesh(mesh1, meshes, orientation):
    # find mesh from list of meshes, that is closest to input mesh
    o = {
    "vertical": 0,
    "horizontal": 1}
    dist = np.Inf
    index = 0
        
    # go through all meshes in list
    for mesh2, i in zip(meshes, range(len(meshes))):

        # if meshes don't correspong, it can not be the nearest mesh      
        if not (mesh_correspondance(mesh1, mesh2, orientation)):
            continue
        
        # calculate distance
        dist_tmp = get_mesh_distance(mesh1, mesh2, orientation)
        
        # if distance is 0: they are the same mesh, skip it
        if dist_tmp == 0:
            continue

        # return distance and index of found nearest mesh
        if dist_tmp < dist:
            dist = dist_tmp
            index = i
    
    return dist, index



def group_corresponding_meshes(corr, corr_tuples, patches,  orientation):
    # create tuples of corresponding meshes
    # needed to calculate midpoints between meshes: midpoints become seeds for void growing
    
    # compare all meshes with each other
    # if the nearest and corresponding mesh is found, group it as tuple
    # return all corresponding meshes as list of tuples
    for i in range(len(patches)):
        mesh1 = patches[i]
        for j in range(i+1, len(patches)):
            mesh2 = patches[j]
            nearest_dist, _  = find_nearest_mesh(mesh1, patches, orientation)
        
      
            if (nearest_dist < get_mesh_distance(mesh1, mesh2, orientation)):
                continue
        
            if (mesh_correspondance(mesh1, mesh2, orientation)):
                color = np.random.rand(3)
                mesh1.paint_uniform_color(color)
                mesh2.paint_uniform_color(color)
            
                # meshes are also returned individually for visualization purposes (could be removed)
                corr.append(mesh1)
                corr.append(mesh2)
                corr_tuples.append((mesh1, mesh2))
    return corr, corr_tuples

    
def create_uniform_pc_from_bb(bb_axis, voxel_size, color=[0,1,0]):
    # create a pointcloud from an axis aligned bounding box
    # each point is equidistant to each other with a given voxel size
    # creates a grid-like, axis-aligned pointcloud with dimensions of bouding box
    
    # create empty pointcloud
    pc = o3d.geometry.PointCloud()

    # transform axis aligned bb into oriented bb (needed for mesh creation)
    bb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bb_axis)

    # create mesh from oriented bounding box (needed for voxel grid creation)
    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(bb)
    mesh = mesh.paint_uniform_color(color)

    # create voxel grid from mesh; get center coordinates of each voxel
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    voxels = vg.get_voxels()
    grid_indexes = [x.grid_index for x in voxels]
    voxel_centers = [vg.get_voxel_center_coordinate(index) for index in grid_indexes]
    
    # fill pointcloud with center coordintes of voxels and return pointcloud
    pc.points.extend(o3d.utility.Vector3dVector(voxel_centers))  
    pc.paint_uniform_color(color)
    return pc


def hull_to_uniform_pc(hull, voxel_size, color):
    # create a pointcloud from a convex hull
    # each point is equidistant to each other with a given voxel size
    # creates a grid-like pointcloud with dimensions of convex hull
    
    # create empty pointcloud
    pc = o3d.geometry.PointCloud()
    
    # create voxel grid from convex hull; get center coordinates of each voxel
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull, voxel_size)
    voxels = vg.get_voxels()
    grid_indexes = [x.grid_index for x in voxels]
    voxel_centers = [vg.get_voxel_center_coordinate(index) for index in grid_indexes]
    
    # fill pointcloud with center coordintes of voxels and return pointcloud
    pc.points.extend(o3d.utility.Vector3dVector(voxel_centers))  
    pc.paint_uniform_color(color)
    return pc


def create_uniform_pc(pcd, voxel_size, color):
    # creates a pointcloud from a pointcloud
    # new pointcloud is quantized with voxel_size => fixed density of output pointcloud
    
    # create empty pointcloud
    pc = o3d.geometry.PointCloud()
    
    # create voxel grid from input pointcloud; get center coordinates of each voxel
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxels = vg.get_voxels()
    grid_indexes = [x.grid_index for x in voxels]
    voxel_centers = [vg.get_voxel_center_coordinate(index) for index in grid_indexes]
    
    # fill pointcloud with center coordintes of voxels and return pointcloud
    pc.points.extend(o3d.utility.Vector3dVector(voxel_centers))  
    pc.paint_uniform_color(color)
    return pc
   

def find_closest_vector_index(input_vector, pc):
    # grid-like pointcloud may not have a point at requested location
    # to do proper knn readius search, find point in grid-pointcloud, that is nearest to abitrary point
    # return the index of the nearest point from grid-like pointcloud to requested point


    # Convert the input vector and vector list to numpy arrays for easier calculations
    input_vector = np.array(input_vector)
    vector_list = np.asarray(pc.points)

    # Calculate the Euclidean distances between the input vector and all vectors in the list
    distances = np.linalg.norm(vector_list - input_vector, axis=1)

    # Find the index of the vector with the smallest distance
    closest_index = np.argmin(distances)

    return closest_index


def knn_search_pointclouds(tree_grid, tree_target, center, radius):
    # do a knn radius search for two pointclouds (octree data structure):
    # grid pointcloud: to represent void space
    # target pointcloud: pointcloud of scanned environment

    [ka, idxa, _] = tree_grid.search_radius_vector_3d(center, radius)
    
    [kb, idxb, _] = tree_target.search_radius_vector_3d(center, radius)
    
    # return number of targets hit k
    # return indexes of hit targets idx
    return ka, kb, idxa, idxb


def get_ids_in_direction(idx_grid, idx_target, coords_grid, coords_target, center, direction):
    # knn radius search returns all indexes around center, regarding of direction: BUT, direction needed for void growing
    # this function returns indexes of points lying in specific direction from center point: up, down, left, right

    # decode direction
    d = {
    "up": (1,False),
    "down": (1,True),
    "left": (0,True),
    "right": (0,False)}
    xory = d[direction][0]
    smaller = d[direction][1]
    
    # collect indexes from grid and target that lie in requested direction
    idx_dir_grid = []
    idx_dir_target = []
    for i in range(len(idx_grid)):
        if smaller:
            if coords_grid[i][xory] < center[xory]:
                idx_dir_grid.append(idx_grid[i])
        else:
            if coords_grid[i][xory] > center[xory]:
                idx_dir_grid.append(idx_grid[i])
    
    for i in range(len(idx_target)):
        if smaller:
            if coords_target[i][xory] < center[xory]:
                idx_dir_target.append(idx_target[i])
        else:
            if coords_target[i][xory] > center[xory]:
                idx_dir_target.append(idx_target[i])
            
            
    return (idx_dir_grid, idx_dir_target)



def grow_void(pcd_grid, pcd_target, initial_seed, initial_set=set(), search_radius=0.2, stop_threshold=2, void_color=[0,0,1]):
    # grow void in grid-pointcloud given an initial seed in void space and target pointcloud
    # already known points can be given as input (initial_set), to avoid unnecessary knn searches

    # create KD-Tree for knn radius search for grid and target pointcloud
    tree_grid = o3d.geometry.KDTreeFlann(pcd_grid)
    tree_target = o3d.geometry.KDTreeFlann(pcd_target)


    # find index of point in grid, that is closest to initial seed
    index = find_closest_vector_index(initial_seed, pcd_grid)  

    # sets to store evaluated points. initialize new_set with found initial index
    old_set = set()
    new_set = set([index])


    # only do knn searches for "not-evaluated-before" points. reduce new_set by "know-before" points (initial_set)
    # save new_set as old set before doing new search round
    different_elements = new_set-initial_set
    old_set = set(new_set) 
    

    # do searches, as long as new void points are found
    while(different_elements):
    

        # go through all new void points and do knn radius search around them             
        for index in different_elements:
            # get coordinates of point in grid at index
            # do a knn radius search
            new_center = pcd_grid.points[index]
            k_grid, k_target, idxu, idxf = knn_search_pointclouds(tree_grid, tree_target, new_center, search_radius)
            
            # save found points as numpy arrays
            coords_grid = np.asarray(pcd_grid.points)[idxu]
            coords_target = np.asarray(pcd_target.points)[idxf]

            
            # split all found points in 4 directions; for grid and target pointclouds
            idx_up_grid, idx_up_target = get_ids_in_direction(idxu, idxf, coords_grid, coords_target, new_center, "up")
            idx_down_grid, idx_down_target = get_ids_in_direction(idxu, idxf, coords_grid, coords_target, new_center, "down")
            idx_left_grid, idx_left_target = get_ids_in_direction(idxu, idxf, coords_grid, coords_target, new_center, "left")
            idx_right_grid, idx_right_target = get_ids_in_direction(idxu, idxf, coords_grid, coords_target, new_center, "right")
            

            # check for all directions, if amount of found points in target pointcloud is below threshold:
            # yes: there is void space in that direction; grid indexes are saved in new_set (void space) and grid-pointcloud is colored accordingly
            # no: it is not void space; do nothing

            if (len(idx_up_target) < stop_threshold):
                new_set.update(idx_up_grid[1:])
                np.asarray(pcd_grid.colors)[idx_up_grid[1:], :] = void_color
            
            if (len(idx_down_target) < stop_threshold):
                new_set.update(idx_down_grid[1:])
                np.asarray(pcd_grid.colors)[idx_down_grid[1:], :] = void_color
            
            if (len(idx_left_target) < stop_threshold):
                new_set.update(idx_left_grid[1:])
                np.asarray(pcd_grid.colors)[idx_left_grid[1:], :] = void_color
            
            if (len(idx_right_target) < stop_threshold):
                new_set.update(idx_right_grid[1:])
                np.asarray(pcd_grid.colors)[idx_right_grid[1:], :] = void_color
        

        # check, what points need to be evaluated next
        # take all found points, reduce by points found in last iteration, reduce by points known before funciton call
        # save new_set as old_set for next iteration
        different_elements = new_set-old_set-initial_set
        old_set = set(new_set) 

    # gather all indexes of void points found (indexes of grid)  
    new_set = new_set.union(initial_set)
    
    # return indexes and colored grid pointcloud
    return pcd_grid, new_set
        


def extract_void_area(pcd_grid, pcd_target, midpoints, known_points = set(),  search_radius=1, void_color=[0,0,1]):
    # grow void for all found midpoints in target pointcloud
    # colorize void area and save point indexes

    for points in midpoints:
        void_area, known_points  = grow_void(pcd_grid, pcd_target, points, known_points, search_radius, void_color=void_color)
        #o3d.visualization.draw_geometries([void_area, pcd_flat]+marker_meshes)

    # return void area as grid-like pointcloud
    return void_area, known_points


        
    