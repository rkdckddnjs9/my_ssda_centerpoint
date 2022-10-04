import open3d as o3d
import numpy as np

def txt_read(path):
    with open(path) as f: 
        a = f.read().splitlines()
    return a

def box_center_to_corner(box_center):
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box_center[0:3]
    h, w, l = box_center[5], box_center[4], box_center[3]
    rotation = box_center[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])
        # return np.array([[1,  0,  0],
        #                  [0,  1,  0],
        #                  [0, 0,  1]])
    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([
                        [np.cos(t), -np.sin(t), 0.0],
                        [np.sin(t), np.cos(t), 0.0],
                        [0.0, 0.0, 1.0]])
        # return np.array([[1,  0,  0],
        #                  [0,  1,  0],
        #                  [0, 0,  1]])

    #R = roty(heading_angle)
    R = rotz(heading_angle)
    #l,w,h = box_size
    l,h,w = box_size
    #w,h,l = box_size
    #w,l,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    #y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # corners_3d[0,:] = corners_3d[0,:] + center[0]
    # corners_3d[1,:] = corners_3d[1,:] + center[1]
    # corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

path = "/home/changwon/detection_task/SSOD/kakao/SSDA/visualization_in_model/"
scene = "merged_1625131846_8720364571"
npy_name = "{}.npy".format(scene) 
txt_name = "{}.txt".format(scene)
#txt_name = "pred_box.txt"
#txt_name = "pred_box_eval.txt"
pc = np.load(path+npy_name)
gt_box_ = txt_read(path + txt_name)[0].split(",")
gt_box = []
for num in range(len(gt_box_)-1):
    gt_box.append(float(gt_box_[num]))

all_cls = int(len(gt_box)/7)
gt_ = []
box_line_ = []
for i in range(all_cls):
    cls = i
    idx = 7 * cls
    gt_box_1 = [[gt_box[idx + 0],gt_box[idx + 1],gt_box[idx + 2]],gt_box[idx + 3],[gt_box[idx + 4],gt_box[idx + 5],gt_box[idx + 6]]]
    gt_.append(gt_box_1)
    box3d_from_label = get_3d_box(gt_box_1[0], gt_box_1[1], gt_box_1[2])
    #print(gt_box_1)

    # gt_box_1 = [gt_box[idx + 4],gt_box[idx + 5],gt_box[idx + 6], gt_box[idx + 0],gt_box[idx + 1],gt_box[idx + 2], gt_box[idx + 3]]
    # gt_.append(gt_box_1)
    # box3d_from_label = box_center_to_corner(gt_box_1)

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    pc_array = np.load(path+npy_name)[:,1:4]
    pc_new = o3d.PointCloud()
    pc_new.points = o3d.Vector3dVector(pc_array)
    pc_new.colors = o3d.Vector3dVector(np.zeros_like(pc_array))


    #open3d.visualization.draw_geometries([pc_new])

    colors = [[1, 0, 1] for _ in range(len(box3d_from_label))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box3d_from_label)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    box_line_.append(line_set)

    # Create a visualization object and window
    vis = o3d.visualization.Visualizer()

pc_new_1 = o3d.voxel_down_sample(pc_new, voxel_size=0.7) #voxelization option
pc_new_1.paint_uniform_color([1, 0.706, 0]) #voxelization된거 색칠하는 방법
coord = o3d.create_mesh_coordinate_frame(1, [0, 0, 0]) #coordinate 생성
#o3d.visualization.draw_geometries([pc_new]+[i for i in box_line_])
o3d.visualization.draw_geometries([pc_new]+[pc_new_1] + [i for i in box_line_] + [coord])
     

print(1)