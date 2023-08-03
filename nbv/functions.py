import open3d as o3d
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import copy as cp
import open3d.core as o3c
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
import time
from scipy.spatial.transform import Rotation
from iteration_utilities import deepflatten
from mpl_toolkits.mplot3d import Axes3D



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
