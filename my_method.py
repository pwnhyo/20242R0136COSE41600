import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

def downsample_pcd(pcd, voxel_size=0.4):
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd

def remove_outlier(pcd, nb_points=6, radius=1.0):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    ror_pcd = pcd.select_by_index(ind)
    return ror_pcd

def segment_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=2000):    
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    return plane_model, inliers

def select_by_index(pcd, inliers, invert=False):
    selected_pcd = pcd.select_by_index(inliers, invert=invert)
    return selected_pcd

def cluster_dbscan(pcd, eps=0.3, min_points=10):   
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels

def visualize_clusters(pcd, labels):
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def filter_clusters(pcd, labels, min_points_in_cluster=5, max_points_in_cluster=40, min_z_value=-1.5, max_z_value=2.5, min_height=0.5, max_height=2.0, max_distance=30.0):
    bboxes = []
    for i in range(labels.max() + 1):
        points = np.asarray(pcd.points)[labels == i]
        if len(points) < min_points_in_cluster or len(points) > max_points_in_cluster:
            continue
        min_z = np.min(points[:, 2])
        max_z = np.max(points[:, 2])
        if min_z < min_z_value or max_z > max_z_value or max_z - min_z < min_height or max_z - min_z > max_height:
            continue
        center = np.mean(points, axis=0)
        if np.linalg.norm(center[:2]) > max_distance:
            continue
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(points, axis=0), max_bound=np.max(points, axis=0))
        bboxes.append(bbox)
    return bboxes

def main(target):
    file_paths = os.listdir(target)
    file_paths.sort()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    view_ctl = vis.get_view_control()
    
    start_time = time.time()
    first_call = True
    geometries = []  # 추가된 bounding box들을 추적

    camera_params = None
    for file_path in file_paths:
        original_pcd = read_pcd(f"{target}/{file_path}")
        downsampled_pcd = downsample_pcd(original_pcd)
        ror_pcd = remove_outlier(downsampled_pcd)
        plane_model, inliers = segment_plane(ror_pcd)
        road_pcd = select_by_index(ror_pcd, inliers)
        non_road_pcd = select_by_index(ror_pcd, inliers, invert=True)
        labels = cluster_dbscan(non_road_pcd)
        non_road_pcd = visualize_clusters(non_road_pcd, labels)
        bboxes = filter_clusters(non_road_pcd, labels)

        if first_call:
            vis.add_geometry(non_road_pcd)
            first_call = False
            while vis.poll_events():
                vis.update_renderer()
                camera_params = view_ctl.convert_to_pinhole_camera_parameters()
                if time.time() - start_time > 10:
                    break
        else:
            vis.clear_geometries()
            vis.add_geometry(non_road_pcd)
        

        view_ctl.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        print("Rendered")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    vis.close()
    vis.destroy_window()


if __name__ == "__main__":
    #main(target="data/01_straight_walk/pcd")
    main(target="data/02_straight_duck_walk/pcd")
    #main(target="data/03_straight_crawl/pcd")
    #main(target="data/04_zigzag_walk/pcd")
    #main(target="data/05_straight_duck_walk/pcd")
    #main(target="data/06_straight_crawl/pcd")
    #main(target="data/07_straight_walk/pcd")
