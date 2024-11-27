import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

def downsample_pcd(pcd, voxel_size=0.2):
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

def cluster_dbscan(pcd, eps=0.6, min_points=5):   
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels

def visualize_clusters(pcd, labels):
    # Create a color array initialized to black
    colors = np.zeros((len(labels), 3))  # Black: [0, 0, 0]
    
    # Set labeled points to red [1, 0, 0]
    colors[labels >= 0] = [0, 0, 1]
    
    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def filter_clusters(pcd, labels, min_points_in_cluster=10, max_points_in_cluster=100, min_z_value=-2, max_z_value=50, min_height=0.2, max_height=20, max_distance=30.0):
    bboxes = []
    colors = np.zeros((len(labels), 3))
    for i in range(labels.max() + 1):
        filtered = 0
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
            if volume < 0.4: # volume check
                if height >= 0.5 * shourlder_width and height >= 0.5 * chest_depth: # human shape check1
                    if height / shourlder_width <= 2 or height / chest_depth <= 2: # human shape check2
                    #if min_z_value <= z_min and z_max <= max_z_value: # z value check
                        #if z_min <= 3: # floating object check
                    #if height >= min_height and height <= max_height: # height check
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0) 
                        bboxes.append(bbox)
                        filtered = 1
        
        
        # Update color based on `filtered` value
        cluster_color = [1, 0, 0] if filtered == 1 else [0, 0, 0]
        colors[cluster_indices] = cluster_color

    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Number of bounding boxes: {len(bboxes)}")
    return bboxes

dangerous_xyz = []

def filter_bboxes(pcd, bboxes, prev_bboxes, flag, param=1):
    if not prev_bboxes:
        return bboxes

    filtered_bboxes = []
    colors = np.zeros((len(pcd.points), 3))  # Default color: black

    def is_near_dangerous(center):
        return any(np.linalg.norm(np.array(dangerous) - np.array(center)) < param for dangerous in dangerous_xyz)

    points = np.asarray(pcd.points)

    for bbox in bboxes:
        bbox_center = np.array(bbox.get_center())
        is_filtered = False

        if flag == 1:
            if is_near_dangerous(bbox_center):
                is_filtered = True

            for prev_bbox in prev_bboxes:
                prev_center = np.array(prev_bbox.get_center())
                if np.linalg.norm(prev_center - bbox_center) < param:
                    is_filtered = True
                    if not is_near_dangerous(bbox_center):
                        dangerous_xyz.append(bbox.get_center())
                    break
        else:
            if is_near_dangerous(bbox_center):
                is_filtered = True

            min_distance = min(np.linalg.norm(np.array(prev_bbox.get_center()) - np.array(bbox_center)) for prev_bbox in prev_bboxes)
            if is_filtered == False and min_distance < 0.1:
                is_filtered = True
                dangerous_xyz.append(bbox.get_center())


        # Get indices of points within the bounding box
        indices = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))

        if not is_filtered:
            filtered_bboxes.append(bbox)
            colors[indices] = [1, 0, 0]  # Red for filtered
        else:
            colors[indices] = [0, 0, 0]  # Black for not filtered

    pcd.colors = o3d.utility.Vector3dVector(colors)  # Update point cloud colors

    print(f"Number of filtered bounding boxes: {len(filtered_bboxes)}")
    print(len(dangerous_xyz))
    return filtered_bboxes



def main(target):
    file_paths = os.listdir(target)
    file_paths.sort()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    vis.get_render_option().point_size = 3.0
    view_ctl = vis.get_view_control()
    
    start_time = time.time()
    first = True

    camera_params = None
    prev_bboxes = []
    frame_cnt = 0
    for file_path in file_paths[:]:
        frame_cnt += 1
        original_pcd = read_pcd(f"{target}/{file_path}")
        downsampled_pcd = downsample_pcd(original_pcd)
        ror_pcd = remove_outlier(downsampled_pcd)
        plane_model, inliers = segment_plane(ror_pcd)
        road_pcd = select_by_index(ror_pcd, inliers)
        non_road_pcd = select_by_index(ror_pcd, inliers, invert=True)
        labels = cluster_dbscan(non_road_pcd)
        non_road_pcd = visualize_clusters(non_road_pcd, labels)

        bboxes_ori = filter_clusters(non_road_pcd, labels)

        if frame_cnt == 3 or frame_cnt % 10 == 0:
            bboxes = filter_bboxes(non_road_pcd, bboxes_ori, prev_bboxes, 1)
            prev_bboxes = bboxes_ori
        else:
            bboxes = filter_bboxes(non_road_pcd,bboxes_ori, prev_bboxes, 0)

        if first:
            vis.add_geometry(non_road_pcd)
            first = False
            for bbox in bboxes:
                vis.add_geometry(bbox)
            while vis.poll_events():
                vis.update_renderer()
                camera_params = view_ctl.convert_to_pinhole_camera_parameters()
                if time.time() - start_time > 3:
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
    main(target="data/04_zigzag_walk/pcd")
    #main(target="data/05_straight_duck_walk/pcd")
    #main(target="data/06_straight_crawl/pcd")
    #main(target="data/07_straight_walk/pcd")
