import open3d as o3d
import numpy as np
import copy
import sys
import argparse

# genarate 2 lists of pointclouds, one with original data, one downsampled version
def load_point_clouds_downsampled(pointclouds, voxel_size=0.0):
    pcds = []
    pcds_down = []
    n_pc = len(pointclouds)
    for i in range(0, n_pc):

        # read pointcloud and estimate normals
        pcd_orig = o3d.io.read_point_cloud(pointclouds[i])
        pcd_orig.estimate_normals()
        pcds.append(pcd_orig)

        # downsample original pointcloud
        pcd_down = pcd_orig.voxel_down_sample(voxel_size=voxel_size)
        pcds_down.append(pcd_down)

    return pcds, pcds_down

# visualize 2 pointclouds with a transformation; copy needed to not alter orientation of original pointcloud
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# create a visualizer and let user pick correspondance points in a pointcloud; return picked points afterwards
def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press q to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

# manual registration of 2 overlapping pointclouds; correspond points used for alignment
def manual_registration(pcd_data):

    # source and target pointcloud are visualized before alingment
    source = pcd_data[1]
    target = pcd_data[0]
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)

    # user needs to pick at least 3 points per pointcloud; number of points in pointcloud need to match for source and target pointcloud
    # return False if not the case
    if not (len(picked_id_source) >= 3 and len(picked_id_target) >= 3):
        return False, None
    if not (len(picked_id_source) == len(picked_id_target)):
        return False, None

    # create array of correspondace points from source and target pointcloud
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement; use inital transformation based on picked points as starting point
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)

    # return True if transformation has been calculated as well as the transformation itself
    return True, reg_p2p.transformation


def pairwise_manual_align(pcds, pcds_down, pointclouds):
    # create list for transforms from pointcloud i to pointcloud i+1
    transforms = []

    # optional: first pointcloud is saved as file for future used; can be commented out
    # save the first pointcloud "as is"; first pointcloud is not transformed and is used as base-frame
    name = pointclouds[0][:len(pointclouds[0])-5]
    name += "%i_prealigned.pcd" % (1)
    o3d.io.write_point_cloud(name, pcds[0])

    # go through all pointclouds and do pairwise alignment based on picked points
    for i in range(len(pcds_down)-1):
        check, tf = manual_registration(pcds_down[i:i+2])

        # check is used to verify if picked points are valid
        # e.g. if user miss-clicked on pointcloud and chose non-matching points, one can can restart procedure by choosing unequal number of points
        # prevents restart of the whole procedure if a single pair-alignment might fail because of invalid chosen points
        while not (check):
            check, tf = manual_registration(pcds_down[i:i+2])

        # transform the pointcloud based on calculated transform
        # if transform is correct, subsequent pointclouds should be aligned with first pointcloud
        pcds[i+1] = pcds[i+1].transform(tf)
        pcds_down[i+1] = pcds_down[i+1].transform(tf)

        # optional: aligned pointclouds are saved as files for future used; can be commented out
        name = pointclouds[i+1][:len(pointclouds[i+1])-5]
        name += "%i_prealigned.pcd" % (i+2)
        o3d.io.write_point_cloud(name, pcds[i+1])

        # append transforms to list and print individual transforms
        transforms.append(tf)
        print("Trasnformation from frame %i to frame %i:" % (i, i+1))
        print(tf)

    return transforms


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
    o3d.io.write_point_cloud(output, pcd_combined)
    o3d.visualization.draw_geometries(pcds)


def main(argv):
    # get arguments from command line
    argParser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=80))
    argParser.add_argument("-i", "--input", nargs="+", help="list of input files", required=True)
    argParser.add_argument("-o", "--output", type=str, default="output.pcd", help="output file name; default=output.pcd")
    argParser.add_argument("-d", "--downsample", type=float, default=0.05, help="voxel size for downsampling; default=0.05")
    argParser.add_argument("-a", "--align", type=bool, default=False, help="set to True if manual pre-alignment is needed; default=False")
    args = argParser.parse_args()

    voxel_size = args.downsample
    pointclouds = args.input
    output_pc = args.output
    align = args.align

    # print chosen parameters
    print("input files: "+", ".join(pointclouds))
    print("output file: %s" % output_pc)
    print("voxel size: %f" % voxel_size)
    print("manual alignment set to: %s" % align)

    # load pointclouds from filr
    pcds, pcds_down = load_point_clouds_downsampled(pointclouds, voxel_size)
    
    # do manual alignment if desired
    if (align):
        transforms = pairwise_manual_align(pcds, pcds_down, pointclouds)
        print("manually evaluated transforms: \n %s" % transforms)
        test = input("Pointclouds loaded and pre-aligned. Press ENTER for ICP registration. Press n + ENTER to cancel\n")
    else:
        test = input("Pointclouds loaded. Press ENTER for ICP registration. Press n + ENTER to cancel\n")
    
    if test == "n":
        return 
    # do ICP registration
    register(pcds, pcds_down, output_pc, voxel_size)


if __name__ == "__main__":
    main(sys.argv)
