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

def detect_ground_plane(pcd):
    """detect ground plane (floor) in pointcloud

    Args:
        pcd (o3d.geometry.PointCloud): input pointcloud

    Returns:
        tuple(np.ndarray, list): plane equation ax+by+cz+d=0 and inlier points
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    
    return plane_model, inliers
    
    
    
    
def compute_plane_transformation(plane1, plane2):
    """
    Computes the transformation between two plane equations of the form ax + by + cz + d = 0.

    Args:
        plane1 (tuple): Coefficients of the first plane equation (a1, b1, c1, d1).
        plane2 (tuple): Coefficients of the second plane equation (a2, b2, c2, d2).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # Extract coefficients of the first plane equation
    a1, b1, c1, d1 = plane1

    # Extract coefficients of the second plane equation
    a2, b2, c2, d2 = plane2

    # Compute the transformation matrix coefficients
    a = a2 * a1 + b2 * b1 + c2 * c1
    b = a2 * d1 - a1 * d2
    c = b2 * d1 - b1 * d2
    d = c2 * d1 - c1 * d2

    transformation_matrix = np.array([[a, 0, 0, b],
                                      [0, a, 0, c],
                                      [0, 0, a, d],
                                      [0, 0, 0, 1]])

    return transformation_matrix

def calculate_floor_alignment_matrix(pcd_ground_plane):
    """Calculates the transformation matrix needed to align the floor of the input pointcloud with the xy-plane

    Args:
        pcd_ground_plane (numpy.ndarray): plane model of the groundplane of the pointcloud. ax+by+cz+d=0

    Returns:
        numpy.ndarray: 4x4 transformation matrix to get alignment of pointcloud floor and xy-plane
    """
    
    
    a, b, c, d = pcd_ground_plane
    d = -d
    # Calculate the normal vector of the plane
    plane_normal = np.array([a, b, c])

    # Find the z-axis (floor normal) of the transformed coordinate system
    z_axis = plane_normal / np.linalg.norm(plane_normal)

    # Define the x-axis (assuming it points in the direction of the original x-axis)
    x_axis = np.array([1, 0, 0])

    # Calculate the y-axis (cross product of z_axis and x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Calculate the new x-axis (cross product of y_axis and z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Construct the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[0, :3] = x_axis
    transform_matrix[1, :3] = y_axis
    transform_matrix[2, :3] = z_axis

    # Find the translation component to align the point cloud with the plane
    centroid_on_plane = -plane_normal * (d / np.linalg.norm(plane_normal))
    transform_matrix[:3, 3] = centroid_on_plane

    return transform_matrix

def detect_boundaries(pcd):
    """detect boundary points in a pointcloud

    Args:
        pcd (o3d.geometry.PointCloud): input pointcloud

    Returns:
        o3d.geometry.PointCloud: output pointcloud containing only the boundary points
    """
    
    tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    
    #compute boundary points. args: search radius, max_nn, angle_threshold
    boundaries, mask = tensor_pcd.compute_boundary_points(0.2, 100, 95)
    
    
    boundaries = boundaries.to_legacy()
    boundaries = boundaries.paint_uniform_color([1,0,0])
    cl, ind = boundaries.remove_radius_outlier(2, 0.2)
    boundaries = boundaries.select_by_index(ind)
    
    return boundaries

def detect_boundary_patches(boundaries):
    """take boundary points and detect patches (detect boundary points that form a hole in pointcloud)

    Args:
        boundaries (o3d.geometry.PointCloud): input pointcloud containing all boundary points

    Returns:
        list(o3d.geometry.TriangleMesh): list of Triangle Meshes representing holes in pointcloud
    """
    
    # estimate and orient normals
    if not boundaries.has_normals():
        boundaries.estimate_normals()
    boundaries.orient_normals_consistent_tangent_plane(30)
    
    # detect bounding boxes from corresponding boundary points
    oboxes = boundaries.detect_planar_patches(normal_variance_threshold_deg=60,
                                                coplanarity_deg=80,
                                                outlier_ratio=0.75,
                                                min_plane_edge_length=1,
                                                min_num_points=5,
                                                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
    
    # create triangle meshes from detected bounding boxes
    patches = []
    for obox in oboxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()        
        patches.append(mesh)
    
    return patches
    
def add_vector_to_point(point, vector, length):
    """adds a vector with a given length to a point in 3d and returns the new point

    Args:
        point (numpy.ndarray): input point in 3d
        vector (numpy.ndarray): vector to be added
        length (float): length of given input vector to be added

    Returns:
        numpy.ndarray: new point in 3d
    """
    # Calculate the new point coordinates
    new_point = point + length * vector

    # Return the new point as a NumPy array
    return new_point

def is_perpendicular(angle, threshold):
    """used to check if a given angle is close to being 90° within a certain threshold

    Args:
        angle (float): calculated angle between two vectors
        threshold (float): angle threshold allowed

    Returns:
        bool: returns True if angle is between 90° +- threshold
    """
    
    if (180 <= angle < 360):
        angle -= 180
        
    if ((90-threshold) <= angle <= (90+threshold)):
        return True
    else:
        return False



def compute_transform(vector1, vector2):
    """compute the translation and rotation needed to transform vector1 to vector2

    Args:
        vector1 (numpy.ndarray): input vector start
        vector2 (numpy.ndarray): input vector target

    Returns:
        tuple(numpy.ndarray, numpy.ndarray): translation and transformation matrix
    """
    
    # Normalize the vectors to unit length
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    # Compute the rotation matrix
    rotation_matrix = Rotation.align_vectors([vector1], [vector2])[0].as_matrix()

    # Compute the translation vector
    translation_vector = vector2 - np.dot(rotation_matrix, vector1)

    return translation_vector, rotation_matrix

def calc_mesh_area_sorted(meshes):
    """
    calculate the area of input meshes and sort in descending order.
    Used to rank the importance of meshes.
    larger mesh => higher importance to cover that hole

    Args:
        meshes (list(o3d.geometry.TriangleMesh)): list of input meshes for which area is calculated (meshes are assumed to be square)

    Returns:
        numpy.ndarray: 2d array, containing index of input list and corresponding area
    """
    
    areas = np.zeros((len(meshes),2))
    for i in range(len(meshes)):
        mesh_bb = meshes[i].get_oriented_bounding_box()
        area = mesh_bb.volume()/mesh_bb.extent[2]
        areas[i][0] = i
        areas[i][1] = area
    sorted_indices = np.argsort(areas[:, 1])[::-1]
    return areas[sorted_indices]
          

def create_box_at_point(point, size=(0.1,0.1,0.1), color=[0,1,0]):
    
    size_x, size_y, size_z = size
    box = o3d.geometry.TriangleMesh.create_box(size_x, size_y, size_z)
    box.paint_uniform_color(color)
    
    box.translate(point, False)
    
    
    return box

def filter_by_normal(pcd, hight_variance = 0.01):
    """Filters a pointcloud according to their normals. Only points with normals facing in x or y direction are kept,
    points with normals in z-direction are neglected.
    Hight of points are set to a random hight in range "hight_variance" defined in parameters.
    (completely flat pointcloud yields errors when calculating bounding box/convex hull;
    thus slight hight differences are chosen)
    Outputs a 2d projection of the pointcloud onto the ground plane.

    Args:
        pcd (o3d.geometry.PointCloud): input pointcloud to be filtered
        hight_variance (float, optional): range of random hight of points. Defaults to 0.01.

    Returns:
        o3d.geometry.PointCloud: 2d projected pointcloud
    """
    
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
    """checks if two vectors are perpendicular to each other within a certain range
        all vectors that are 90° +- threshold to each others are considered perpendicular

    Args:
        v1 (numpy.ndarray): input vector1
        v2 (numpy.ndarray): input vector2
        threshold (float): threshold range in degrees

    Returns:
        bool: True: if vectors are perpendicular in defined range
    """
    
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
    
def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    # Normalize the vectors to unit length
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    return np.degrees(angle)

def rotation_matrix_to_axis_angle(matrix):
    """return the angle of rotation around xyz axis from rotation matrix

    Args:
        matrix (numpy.ndarray): input rotation matrix

    Returns:
        numpy.ndarray: array containing rotation around xyz-axis in degrees
    """ 
    r = Rotation.from_matrix(matrix)
    return r.as_euler('xyz', degrees=True)


def invert_normalized_vector(vector):
    """invert the direction of a normalized vector

    Args:
        vector (numpy.ndarray): input vector to be inverted

    Returns:
        numpy.ndarray: output vector pointing in opposite direction
    """
    inverted_vector = []
    for component in vector:
        inverted_component = -component
        inverted_vector.append(inverted_component)
    return inverted_vector



def check_vector_similar_direction(vec1, vec2):
    """check wether two vectors point in a similar direction

    Args:
        vec1 (numpy.ndarray): input vector 1
        vec2 (numpy.ndarray): input vector 2

    Returns:
        bool: returns True if both vectors point in similar direction
    """
    
    # Normalize the vectors
    normalized_vec1 = np.array(vec1) / np.linalg.norm(vec1)
    normalized_vec2 = np.array(vec2) / np.linalg.norm(vec2)
    
    # Calculate the dot product
    dot_product = np.dot(normalized_vec1, normalized_vec2)
    
    if (dot_product > 0):
        return True
    else:
        return False


def sort_xd_list(list, axis, order="ascending"):
    """Sort a multi-dimensional list according to a specific axis

    Args:
        list (list): input list to be sorted
        axis (int): axis along which the list should be sorted
        order (str, optional): choose between ascending or descending order. Defaults to "ascending".

    Returns:
        list: sorted list
    """
    ord = {"ascending" : True,
             "descending" : False}
    list.sort(key=lambda x:x[axis])
    
    if ord[order]:
        return list
    else:
        list.reverse()
        return list
    

def find_midpoint_between_planes(plane1, plane2):
    """calculate midpoint between two planes

    Args:
        plane1 (o3d.geometry.TriangleMesh): input plane1
        plane2 (o3d.geometry.TriangleMesh): input plane2

    Returns:
        numpy.ndarray: 3d coordinates of the midpoint between the two planes
    """

    center1 = plane1.get_center()
    center2 = plane2.get_center()

    vec_1_2 = center2 - center1

    midpoint = center1 + vec_1_2 / 2
    return midpoint

def find_midpoints(corr_tuples):
    """find midpoints for a list of corresponding meshes (walls)
return midpoints as well as markers for visualization

    Args:
        corr_tuples (list[tuple(o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh)]): list of tuples; each tuples contains corresponding planes

    Returns:
        tuple(numpy.ndarray, o3d.geometry.TriangleMesh): tuple containing the coordinates of the midpoint and a box (triangle mesh) at that point for visualization
    """
    
    midpoints = []
    marker_meshes = []

    for tup in corr_tuples:
        midpoint = find_midpoint_between_planes(tup[0], tup[1])
    
    
        midpoints.append(midpoint)
        marker_meshes.append(create_box_at_point(midpoint))
    return midpoints, marker_meshes


def divide_meshes_hor_ver(meshes):
    """divides meshes (that represent walls) into horizontal and vertical meshes

    Args:
        meshes (list[o3d.geometry.TriangleMesh]): list of meshes that shall be divided into horizontal and vertical

    Returns:
        tuple(o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh): tuple(horizontal meshes, vertical meshes)
    """
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
    """calculates the distance between two meshes in the direction defined by parameter orientation
    points used for measurements are centers of bounding boxes

    Args:
        mesh1 (o3d.geometry.TriangleMesh): input mesh1
        mesh2 (o3d.geometry.TriangleMesh): input mesh2
        orientation (string): orientation; either "horizontal" or "vertical"

    Returns:
        float: distance calculated
    """
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


def mesh_correspondence(mesh1, mesh2, orientation):
    """function to check wether two parallel meshes are "corresponding" to each other
    => both meshes probably belong to opposing walls

    Args:
        mesh1 (o3d.geometry.TriangleMesh): input mesh1
        mesh2 (o3d.geometry.TriangleMesh): input mesh2
        orientation (string): orientation; either "horizontal" or "vertical"

    Returns:
        bool: True: if both meshes are considered to be corresponding
    """
    
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
    
    # calculate distance between parallel meshes
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
    """find a mesh from a list of meshes, that is closest to given input mesh1
    distance is evaluated according to given orientation (distance in x or y direction)
    If meshes do not correspond to each other, distance is not evaluated.
    If distance is =0, it is not taken as nearest mesh.

    Args:
        mesh1 (o3d.geometry.TriangleMesh): input mesh, for which nearest mesh shall be found
        meshes (list(o3d.geometry.TriangleMesh)): list of meshes to be evaluated
        orientation (string): orientation; either "horizontal" or "vertical"

    Returns:
        tuple(float, int): tuple: distance and index of mesh in list found to be nearest
    """
    # find mesh from list of meshes, that is closest to input mesh
    o = {
    "vertical": 0,
    "horizontal": 1}
    dist = np.Inf
    index = 0
        
    # go through all meshes in list
    for mesh2, i in zip(meshes, range(len(meshes))):

        # if meshes don't correspong, it can not be the nearest mesh      
        if not (mesh_correspondence(mesh1, mesh2, orientation)):
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
    """creates tuples of corresponding meshes

    Args:
        corr (list): all meshes that have a correspondence; either empty or filled with prior known correspondences
        corr_tuples (list[tuple]): tuples of corresponding meshes; either empty or filled with prior known tuples
        patches (o3d.geometry.TriangleMesh): input meshes to be evaluated
        orientation (string): orientation; either "horizontal" or "vertical"

    Returns:
        tuple(list, list[tuple]): tuple(list of all meshes that have a correspondence, list of corresponding tuples)
    """
    
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
        
            if (mesh_correspondence(mesh1, mesh2, orientation)):
                color = np.random.rand(3)
                mesh1.paint_uniform_color(color)
                mesh2.paint_uniform_color(color)
            
                # meshes are also returned individually for visualization purposes (could be removed)
                corr.append(mesh1)
                corr.append(mesh2)
                corr_tuples.append((mesh1, mesh2))
    return corr, corr_tuples

    
def create_uniform_pc_from_bb(bb_axis, voxel_size=0.1, color=np.asarray([0,1,0])):
    """create a pointcloud from an axis aligned bounding box
    each point is equidistant to each other with a given voxel size
    creates a grid-like, axis-aligned pointcloud with dimensions of bounding box

    Args:
        bb_axis (open3d.geometry.AxisAlignedBoundingBox): axis aligned bounding box, for which grid-like pointcloud should be created
        voxel_size (float): voxel size, defining distance from point to point
        color (numpy.ndarray, optional): color of the output pointcloud. Defaults to [0,1,0].

    Returns:
        o3d.geometry.PointCloud: output pointcloud with grid-like structure
    """
  
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
    
    # fill pointcloud with center coordinates of voxels and return pointcloud
    pc.points.extend(o3d.utility.Vector3dVector(voxel_centers))  
    pc.paint_uniform_color(color)
    return pc


def hull_to_uniform_pc(hull, voxel_size, color):    
    """create a pointcloud from a convex hull
    each point is equidistant to each other with a given voxel size
    creates a grid-like pointcloud with dimensions of convex hull 

    Args:
        hull (open3d.geometry.TriangleMesh): convex hull in form of a triangle mesh
        voxel_size (float): voxel size, defining distance from point to point
        color (numpy.ndarray): color of output pointcloud

    Returns:
        o3d.geometry.PointCloud: output pointcloud with grid-like structure
    """
    # create empty pointcloud
    pc = o3d.geometry.PointCloud()
    
    # create voxel grid from convex hull; get center coordinates of each voxel
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull, voxel_size)
    voxels = vg.get_voxels()
    grid_indexes = [x.grid_index for x in voxels]
    voxel_centers = [vg.get_voxel_center_coordinate(index) for index in grid_indexes]
    
    # fill pointcloud with center coordinates of voxels and return pointcloud
    pc.points.extend(o3d.utility.Vector3dVector(voxel_centers))  
    pc.paint_uniform_color(color)
    return pc


def create_uniform_pc(pcd, voxel_size, color):
    """creates a pointcloud from a pointcloud
    new pointcloud is quantized with voxel_size => fixed density of output pointcloud

    Args:
        pcd (o3d.geometry.PointCloud): input pointcloud that should be voxelized
        voxel_size (float): voxel size, defining distance from point to point
        color (numpy.ndarray): color of output pointcloud

    Returns:
        o3d.geometry.PointCloud: output pointcloud with grid-like structure
    """
    
    
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
    """grid-like pointcloud may not have a point at requested location
    to do proper knn readius search, find point in grid-pointcloud, that is nearest to arbitrary point.
    Return the index of the nearest point from grid-like pointcloud to requested point


    Args:
        input_vector (numpy.ndarray): point for which index of nearest point in grid-like pointcloud shall be found
        pc (o3d.geometry.PointCloud): grid-like input pointcloud which is searched for nearest point

    Returns:
        int: index of closest point from input pointcloud to input point
    """
    
    # Convert the input vector and vector list to numpy arrays for easier calculations
    input_vector = np.array(input_vector)
    vector_list = np.asarray(pc.points)

    # Calculate the Euclidean distances between the input vector and all vectors in the list
    distances = np.linalg.norm(vector_list - input_vector, axis=1)

    # Find the index of the vector with the smallest distance
    closest_index = np.argmin(distances)

    return closest_index


def knn_search_pointclouds(tree_grid, tree_target, center, radius):
    """do a knn radius search for two pointclouds (given as octree data structure)
        find all points in both pointclouds which distance to "center" is equal or lower than "radius"

    Args:
        tree_grid (open3d.geometry.Octree): grid-like pointcloud
        tree_target (open3d.geometry.Octree): arbitrary pointcloud
        center (numpy.ndarray): center point which surroundings should be searched
        radius (numpy.ndarray): radius to be searched

    Returns:
        tuple(int, int, list[int], list[int]): tuple containing: number of found points in pointcloud a and b; indexes of found points in pointcloud a and b
    """
    # do a knn radius search for two pointclouds (octree data structure):
    # grid pointcloud: to represent void space
    # target pointcloud: pointcloud of scanned environment

    [ka, idxa, _] = tree_grid.search_radius_vector_3d(center, radius)
    
    [kb, idxb, _] = tree_target.search_radius_vector_3d(center, radius)
    
    # return number of targets hit k
    # return indexes of hit targets idx
    return ka, kb, idxa, idxb


def get_ids_in_direction(idx_grid, idx_target, coords_grid, coords_target, center, direction):
    """knn radius search returns all indexes around center, regarding of direction: BUT, direction needed for void growing
        this function returns indexes of points lying in specific direction from center point: up, down, left, right

    Args:
        idx_grid (list[int]): list of indexes found by knn radius search in grid pointcloud
        idx_target (list[int]): list of indexes found by knn radius search in target pointcloud
        coords_grid (numpy.ndarray): 2d array containing coordinates of points found by knn radius search
        coords_target (numpy.ndarray): 2d array containing coordinates of points found by knn radius search
        center (numpy.ndarray): coordinates of center point of knn radius search
        direction (string): direction, in which points should be classified. either "up", "down", "left" or "right"

    Returns:
        _type_: _description_
    """
  

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
    """grow void in grid-pointcloud given an initial seed in void space and target pointcloud
        already known points can be given as input (initial_set), to avoid unnecessary knn searches

    Args:
        pcd_grid (o3d.geometry.PointCloud): grid-like pointcloud, in which void grows
        pcd_target (o3d.geometry.PointCloud): target pointcloud, which is searched for void space
        initial_seed (numpy.ndarray): coordinates of initial void seed
        initial_set (set, optional): indexes of points that have been classified as void space in prior iterations. Defaults to set().
        search_radius (float, optional): radius for knn radius search. Defaults to 0.2.
        stop_threshold (int, optional): number of found points in direction must be smaller than threshold, in order to define the direction as void space . Defaults to 2.
        void_color (numpy.ndarray, optional): color of void-space in output pointcloud. Defaults to [0,0,1].

    Returns:
        tuple(o3d.geometry.PointCloud, set(int)): tuple(gird-like pointcloud(void space colorized), indexes of points classified as void space)
    """
    

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
    """grow void for all found midpoints in target pointcloud,
        colorize void area and save point indexes

    Args:
        pcd_grid (o3d.geometry.PointCloud): grid-like pointcloud in which void space is colorized
        pcd_target (o3d.geometry.PointCloud): target pointcloud in which is searched for void space
        midpoints (list(numpy.ndarray)): list of midpoints that serve as initial seeds for void growing
        known_points (set, optional): set of indexes of possibly prior-know void space points. Defaults to set().
        search_radius (int, optional): radius for knn radius search. Defaults to 1.
        void_color (list, optional): color of void space in output pointcloud. Defaults to [0,0,1].

    Returns:
        tuple(o3d.geometry.PointCloud, set(int)): tuple(gird-like pointcloud(void space colorized), indexes of points classified as void space)
    """


    for points in midpoints:
        void_area, known_points  = grow_void(pcd_grid, pcd_target, points, known_points, search_radius, void_color=void_color)
        #o3d.visualization.draw_geometries([void_area, pcd_flat]+marker_meshes)

    # return void area as grid-like pointcloud
    return void_area, known_points


        

def count_hits_ray_cast(scene_answer, cam_normal, dist_thresh, angle_thresh):
    """Counts the number of valid hits in a ray-casting scene.
    Takes constraints like distance to surface and surface normal into account.
    Returns percentage of valid hits.

    Args:
        scene_answer (open3d.t.geometry.RaycastingScene): input ray-casting scene. (simulated pinhole camera)
        cam_normal (numpy.ndarray): orientation of the simulated pinhole camera from ray-casting scene
        dist_thresh (tuple(float, float)): tuple containing upper and lower threshold for which a hit is counted as valid
        angle_thresh (float): ideal angle between surface normal and cam normal: 90°. angle_thresh defines maximum derivation from 90° to be still counted as valid hit

    Returns:
        float: percentage of valid hits in ray-casting scene
    """
    
    # extract surface normals and distances from ray-casting scene
    cast_normals = scene_answer['primitive_normals'].numpy()
    cast_distances = scene_answer['t_hit'].numpy()
    
    dist_lower = dist_thresh[0]
    dist_upper = dist_thresh[1]
    n_valid_hits = 0
    
    # flatten scene answers from n_pixesl x n_pixels to 1 x n_pixels²
    flattened_dist_array = cast_distances.flatten()
    n_pixels = len(flattened_dist_array)
    flattened_normal_array = cast_normals.reshape(n_pixels,3)
    

    # count valid hits: only if in distance bounds and if surface normal does not diverge more then angle_thresh degrees
    for i in range(n_pixels):
        
        if (flattened_dist_array[i] >= dist_lower) & (flattened_dist_array[i] <= dist_upper):
            pass
        else:
            continue
        
        point_normal = flattened_normal_array[i]
        angle = angle_between_vectors(point_normal, cam_normal)
        
        if is_perpendicular(angle, angle_thresh):
            continue
        else:
            n_valid_hits += 1
        
    
    return n_valid_hits/n_pixels

def rank_all_views(hole_patches, dist_thresh, angle_thresh, camera_fov=90):
    """Calculate percentage of valid hits for all potential views.
    Returns percentage of valid hits, camera position and orientation as well as ray-casting scene answers.

    Args:
        hole_patches (list[o3d.geometry.TriangleMesh]): list of triangle meshes representing the holes in a pointcloud
        dist_thresh (tuple(float, float)): tuple containing upper and lower bound for determining valid hits. valid if: lower bound <= distance hit <= upper bound
        angle_thresh (float): ideal angle between surface normal and cam normal: 90°. angle_thresh defines maximum derivation from 90° to be still counted as valid hit
        camera_fov(int): camera field of view for pinhole camera in ray-casting scene. Default: 90°

    Returns:
        list: returns percentage of valid hits, with corresponding camera position and orientation as well as ray-casting scene answers
    """
    
    # create empty ray-casting scene and variables
    scene = o3d.t.geometry.RaycastingScene()
    hole_centers = []
    hole_normals = []
    cam_positions = []
    cam_normals = []
    hit_areas = []
    scene_answers = []
    dist_threshes = [dist_thresh]*len(hole_patches)
    angle_threshes = [angle_thresh]*len(hole_patches)
    
    
    # add meshes that represent holes to ray-casting scene
    for hole in hole_patches:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hole))
        center = hole.get_center()
        normal = np.asarray(hole.vertex_normals)[0]
        cam_pos = add_vector_to_point(center, normal, 5)
        cam_pos[2] = 0.5
        center[2] = 0.5
        hole_centers.append(center)
        hole_normals.append(normal)
        cam_positions.append(cam_pos)
        cam_normal = (center-cam_pos)
        cam_normals.append(cam_normal)
    
    
    # create rays representing virtual pinhole camera for all camera positions and orientations
    for i in range(len(hole_patches)):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                fov_deg=camera_fov,
                center=hole_centers[i],
                eye=cam_positions[i],
                up=[0, 0, -1],
                width_px=500,
                height_px=500)
        ans = scene.cast_rays(rays)
        scene_answers.append(ans)
    
    
    # calculate percentage of valid hits for all scenes in parallel
    with Pool(mp.cpu_count()) as p:
        hit_areas = p.starmap(count_hits_ray_cast, zip(scene_answers, cam_normals, dist_threshes, angle_threshes))
  
    
    
    return list(zip(hit_areas, cam_positions, cam_normals, scene_answers))

