import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

def downsample_pcd(pcd, voxel_size=0.3):
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd

def remove_outlier(pcd, nb_points=4, radius=1.0):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    ror_pcd = pcd.select_by_index(ind)
    return ror_pcd

def segment_plane(pcd, distance_threshold=0.3, ransac_n=5, num_iterations=2000):    
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    return plane_model, inliers

def select_by_index(pcd, inliers, invert=False):
    selected_pcd = pcd.select_by_index(inliers, invert=invert)
    return selected_pcd

def cluster_dbscan(pcd, eps=0.5, min_points=5):   
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels

def visualize_clusters(pcd, labels):
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def filter_clusters(pcd, labels, min_points_in_cluster=7, max_points_in_cluster=30, min_z_value=-2, max_z_value=3, min_height=0.2, max_height=1.8, max_distance=30.0):
    bboxes = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)

            x_values = points[:, 0]  # X값 추출
            y_values = points[:, 1]  # Y값 추출
            z_values = points[:, 2]  # Z값 추출

            x_min = x_values.min()
            x_max = x_values.max()
            y_min = y_values.min()
            y_max = y_values.max()
            z_min = z_values.min()
            z_max = z_values.max()
            
            shourlder_width = x_max - x_min
            chest_depth = y_max - y_min
            height = z_max - z_min

            volume = np.prod(cluster_pcd.get_axis_aligned_bounding_box().get_extent())
            if volume < 0.3 and volume >= 0.03: # volume check
                if height >= 0.7 *shourlder_width and height >= 0.7 *chest_depth: # human shape check1
                    if height / shourlder_width <= 1.7 or height / chest_depth <= 1.7: # human shape check2
                        if min_z_value <= z_min and z_max <= max_z_value: # z value check
                            if z_min <= 3: # floating object check
                                if height >= min_height and height <= max_height: # height check
                                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                                    bbox.color = (1, 0, 0) 
                                    bboxes.append(bbox)
    print(f"Number of bounding boxes: {len(bboxes)}")
    return bboxes

def main(target):
    file_paths = os.listdir(target)
    file_paths.sort()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    view_ctl = vis.get_view_control()
    
    start_time = time.time()
    first = True

    camera_params = None
    for file_path in file_paths[:]:
        original_pcd = read_pcd(f"{target}/{file_path}")
        downsampled_pcd = downsample_pcd(original_pcd)
        ror_pcd = remove_outlier(downsampled_pcd)
        plane_model, inliers = segment_plane(ror_pcd)
        road_pcd = select_by_index(ror_pcd, inliers)
        non_road_pcd = select_by_index(ror_pcd, inliers, invert=True)
        labels = cluster_dbscan(non_road_pcd)
        non_road_pcd = visualize_clusters(non_road_pcd, labels)
        bboxes = filter_clusters(non_road_pcd, labels)

        if first:
            vis.add_geometry(non_road_pcd)
            first = False
            for bbox in bboxes:
                vis.add_geometry(bbox)
            while vis.poll_events():
                vis.update_renderer()
                camera_params = view_ctl.convert_to_pinhole_camera_parameters()
                if time.time() - start_time > 10:
                    break
        else:
            vis.clear_geometries()
            vis.add_geometry(non_road_pcd)
            for bbox in bboxes:
                vis.add_geometry(bbox)
        

        view_ctl.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        print("Rendered")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    vis.close()
    vis.destroy_window()


if __name__ == "__main__":
    #main(target="test_data/")
    #main(target="data/01_straight_walk/pcd")
    #main(target="data/02_straight_duck_walk/pcd")
    #main(target="data/03_straight_crawl/pcd")
    #main(target="data/04_zigzag_walk/pcd")
    #main(target="data/05_straight_duck_walk/pcd")
    #main(target="data/06_straight_crawl/pcd")
    main(target="data/07_straight_walk/pcd")
