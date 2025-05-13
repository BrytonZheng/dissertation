import math
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.io import savemat

CONST_MAT_NAME = "data.mat"


def pixel_to_world_ground(px, py, camera_params):
    """
    将像素坐标转换为世界坐标系中的地面交点，满足：
    - 越靠近主点(cx,cy)，交点距离越远
    - 主点本身返回无穷远
    """
    # 解析相机参数
    cam_pos = np.array(camera_params["pos"])
    cx, cy = camera_params["cx"], camera_params["cy"]
    fx, fy = camera_params["fx"], camera_params["fy"]
    height = -cam_pos[2]  # 假设地面z=0，相机高度为pos.z的绝对值

    # 计算像素归一化偏移（光学投影原理）
    dir_x = (px - cx) / fx
    dir_y = (py - cy) / fy  # 归一化投影方向

    # 计算俯仰角
    pitch = np.arctan(dir_y)

    # 处理主点极限情况
    min_pitch = np.radians(0.001)
    if abs(pitch) < min_pitch:
        if np.sqrt(dir_x ** 2 + dir_y ** 2) < 1e-6:  # 主点附近
            return (float('inf'), float('inf'), 0), float('inf')
        else:
            pitch = np.sign(pitch) * min_pitch

    # 计算地面交点（相机坐标系）
    distance = height / np.tan(pitch)
    x_cam = distance * dir_x
    y_cam = -distance  # 关键修正：地面点在相机坐标系中 -Z 方向

    # 转换到世界坐标系（考虑相机旋转）
    rot = camera_params["rot"]
    R = euler_to_rotation_matrix(rot)
    ground_point_cam = np.array([x_cam, y_cam, 0])  # 确保地面点的Z=0
    ground_point = R @ ground_point_cam + cam_pos

    ground_point[2] = 0  # 强制 z=0

    # 使用相机 yaw 角来对齐世界坐标系
    yaw = camera_params["rot"][2] + 0.03  # 单位是度

    # 对 ground_point[:2] 进行反向旋转
    aligned_point = rotate_point_around_z(ground_point[:2], -yaw)
    return aligned_point


def world_ground_to_pixel(xw_aligned, yw_aligned, camera_params):
    """
    从“以相机朝向为准的对齐世界坐标系”中恢复像素坐标
    """
    cam_pos = np.array(camera_params["pos"])
    cx, cy = camera_params["cx"], camera_params["cy"]
    fx, fy = camera_params["fx"], camera_params["fy"]
    rot = camera_params["rot"]

    # 将对齐后的坐标，正向旋转回去，得到原始世界坐标
    yaw = rot[2] + 0.03
    xw, yw = rotate_point_around_z((xw_aligned, yw_aligned), yaw)

    # 后续逻辑和原来一样
    ground_world = np.array([xw, yw, 0])
    R = euler_to_rotation_matrix(rot)
    R_inv = R.T

    vec = ground_world - cam_pos
    ground_cam = R_inv @ vec

    xc, yc, zc = ground_cam
    px = fx * (xc / -yc) + cx
    py = fy * (zc / -yc) + cy

    return px, py


def euler_to_rotation_matrix(angles_rad):
    roll, pitch, yaw = angles_rad
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


def rotate_point_around_z(point, angle_rad):
    x, y = point
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    return np.array([x_rot, y_rot])


def get_camera_params(dir):
    with open(dir, "r") as jf:
        json_data = json.load(jf)
        json_camera = json_data["camera"]
    return json_camera


def test_main():
    print("TESTING")
    json_dir = "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/"
    camera_dump_dir = "DumpSettings.json"

    # get camera settings
    json_camera = get_camera_params(json_dir + camera_dump_dir)

    px, py = 500, 500
    wx, wy = pixel_to_world_ground(px, py, json_camera)
    print("Pixel to World:", px, py, "->", wx, wy)

    wxx, wyy = wx, wy
    pxx, pyy = world_ground_to_pixel(wxx, wyy, json_camera)
    print("World to Pixel:", wxx, wyy, "->", pxx, pyy)


def main(json_dir = "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220"):
    # json_dir = "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220"
    camera_dump_dir = "DumpSettings.json"
    frame_file = "CameraInfo.json"
    frame_rate = 5
    v_window = 3
    a_window = 1
    lateral_maneuver_past_window = 20
    lateral_maneuver_future_window = 20
    lateral_maneuver_change_threshold = 3
    longitudinal_maneuver_past_window = 15
    longitudinal_maneuver_future_window = 25
    same_lane_threshold = 3
    neighbor_consider_threshold = 90
    scale = 0.3048

    # get camera settings
    json_camera = get_camera_params(os.path.join(json_dir, camera_dump_dir))

    # get traj
    traj_data = {}
    cnt = int(1)
    id_dict = {}
    id_iota = 1
    for subfolder in sorted([f for f in os.listdir(json_dir) if f.isdigit()], key = int):
        # get frame json
        sf = str(subfolder)
        path = os.path.join(json_dir, sf, frame_file)

        if os.path.isfile(path):
            with open(path, "r") as jf:
                frame_data = json.load(jf)
                for obj in frame_data["bboxes"] + frame_data["bboxesCulled"]:
                    if obj["type"] == 4:
                        bbox = obj["bbox"]
                        px, py = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                        coord = pixel_to_world_ground(px, py, json_camera)
                        plt.scatter(-coord[1], -coord[0], s = 1)
                    elif obj["type"] == 6:
                        bbox = obj["bbox"]
                        px, py = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                        coord = pixel_to_world_ground(px, py, json_camera)
                        if obj["id"] not in id_dict:
                            id_dict[obj["id"]] = id_iota
                            traj_data[id_dict[obj["id"]]] = {}
                            for i in range(11):
                                traj_data[id_dict[obj["id"]]][i] = []
                            id_iota += 1

                        vehicle_id = id_dict[obj["id"]]
                        if vehicle_id == 1:
                            print(cnt * 12 + 12, px, py, coord)

                        traj_data[vehicle_id][0].append(1)
                        traj_data[vehicle_id][1].append(vehicle_id)
                        traj_data[vehicle_id][2].append(cnt)
                        traj_data[vehicle_id][3].append(float(-coord[1]))
                        traj_data[vehicle_id][4].append(float(-coord[0]))
                        traj_data[vehicle_id][7].append(1)
                        traj_data[vehicle_id][8].append(2)
        else:
            print("[ERROR]Get json file", path)
        cnt += 1

    # calculate velocity and acceleration
    for _, traj in traj_data.items():
        total_frame = len(traj[0])
        for i in range(1, v_window):
            x1, y1 = traj[3][0], traj[4][0]
            x2, y2 = traj[3][i], traj[4][i]
            v = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * frame_rate / i
            traj[5].append(v)
        for i in range(v_window, total_frame):
            x1, y1 = traj[3][i - v_window], traj[4][i - v_window]
            x2, y2 = traj[3][i], traj[4][i]
            v = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * frame_rate / v_window
            traj[5].append(v)
        # traj[5] = savgol_filter(traj[5], window_length = 7, polyorder = 2).tolist()
        traj[5].insert(0, traj[5][0])
        for i in range(total_frame - a_window):
            v1 = traj[5][i]
            v2 = traj[5][i + a_window]
            a = (v2 - v1) * frame_rate / a_window
            traj[6].append(a)
        for i in range(total_frame - a_window, total_frame - 1):
            to_end = total_frame - i - 1
            v1 = traj[5][i]
            v2 = traj[5][i + to_end]
            a = (v2 - v1) * frame_rate / to_end
            traj[6].append(a)
        # traj[6] = savgol_filter(traj[6], window_length = 7, polyorder = 2).tolist()
        traj[6].append(traj[6][-1])

    # calculate lateral maneuver TODO: 要改xy的话这里也要改
    for _, traj in traj_data.items():
        total_frame = len(traj[0])
        for i in range(total_frame):
            upper_bound = min(total_frame - 1, i + lateral_maneuver_future_window)
            lower_bound = max(0, i - lateral_maneuver_past_window)
            if (traj[4][upper_bound] - traj[4][i] > lateral_maneuver_change_threshold or
                    traj[4][i] - traj[4][lower_bound] > lateral_maneuver_change_threshold):
                traj[9].append(3)
            elif (traj[4][upper_bound] - traj[4][i] < -lateral_maneuver_change_threshold or
                  traj[4][i] - traj[4][lower_bound] < -lateral_maneuver_change_threshold):
                traj[9].append(2)
            else:
                traj[9].append(1)

    # calculate longitudinal maneuver
    for _, traj in traj_data.items():
        total_frame = len(traj[0])
        for i in range(total_frame):
            upper_bound = min(total_frame - 1, i + longitudinal_maneuver_future_window)
            lower_bound = max(0, i - longitudinal_maneuver_past_window)
            if upper_bound == i or lower_bound == i:
                traj[10].append(1)
            else:
                v_history = abs(traj[3][i] - traj[3][lower_bound]) / (i - lower_bound) * frame_rate
                v_future = abs(traj[3][upper_bound] - traj[3][i]) / (upper_bound - i) * frame_rate
                if v_future / v_history < 0.8:
                    traj[10].append(2)
                elif v_future / v_history > 1.25:
                    traj[10].append(3)
                else:
                    traj[10].append(1)

    # calculate grid
    time_frames = []
    for _, traj in traj_data.items():
        total_frame = len(traj[0])
        for i in range(11, 50):
            traj[i] = [0] * total_frame
        for i in range(total_frame):
            time_frames.append(traj[2][i])
    time_frames = np.unique(time_frames)

    traj_times = {}
    for _, traj in traj_data.items():
        total_frame = len(traj[0])
        for i in range(total_frame):
            t = traj[2][i]
            traj_row = []
            for j in range(11):
                traj_row.append(traj[j][i])

            if t not in traj_times:
                traj_times[t] = []
            traj_times[t].append(traj_row)

    for _, traj in traj_data.items():
        total_frame = len(traj[0])
        for i in range(total_frame):
            t = traj[2][i]
            cur_vehicles = traj_times[t]
            same_lane, left_lane, right_lane = [], [], []
            for vehicle in cur_vehicles:
                dy = vehicle[4] - traj[4][i]
                if abs(dy) < same_lane_threshold / 2:
                    same_lane.append(vehicle)
                elif same_lane_threshold / 2 < dy < same_lane_threshold * 1.5:
                    left_lane.append(vehicle)
                elif -same_lane_threshold * 1.5 < dy < -same_lane_threshold / 2:
                    right_lane.append(vehicle)
            if len(left_lane) != 0:
                for vehicle in left_lane:
                    dx = vehicle[3] - traj[3][i]
                    if abs(dx) < neighbor_consider_threshold:
                        target_grid = 1 + round((dx + neighbor_consider_threshold) / 15)
                        traj[10 + target_grid][i] = vehicle[1]
            for vehicle in same_lane:
                dx = vehicle[3] - traj[3][i]
                if abs(dx) < neighbor_consider_threshold and dx != 0:
                    target_grid = 14 + round((dx + neighbor_consider_threshold) / 15)
                    traj[10 + target_grid][i] = vehicle[1]
            if len(right_lane) != 0:
                for vehicle in right_lane:
                    dx = vehicle[3] - traj[3][i]
                    if abs(dx) < neighbor_consider_threshold:
                        target_grid = 27 + round((dx + neighbor_consider_threshold) / 15)
                        traj[10 + target_grid][i] = vehicle[1]

    # generate mat file
    mat_traj = []
    for traj in traj_data.values():
        total_frame = len(traj[0])
        for i in range(15, total_frame - 1):
            traj_row = []
            for j in range(50):
                traj_row.append(traj[j][i])
            traj_row[3], traj_row[4] = np.array(traj_row[4]) / scale, np.array(traj_row[3]) / scale
            mat_traj.append(traj_row)
    max_vid = 0
    for traj in mat_traj:
        vehicle_id = traj[1]
        max_vid = max(max_vid, vehicle_id)
    mat_track = [[] for _ in range(max_vid + 1)]
    for traj in traj_data.values():
        vehicle_id = traj[1][0]
        for i in range(2, 50):
            mat_track[vehicle_id - 1].append(traj[i])
        mat_track[vehicle_id - 1][1], mat_track[vehicle_id - 1][2] = (
            np.array(mat_track[vehicle_id - 1][2]) / scale, np.array(mat_track[vehicle_id - 1][1]) / scale)

    mat_dir = os.path.join(json_dir, CONST_MAT_NAME)
    savemat(mat_dir, {"traj": mat_traj, "tracks": mat_track})
    # xx = np.array(traj_data[2][3])
    # yy = np.array(traj_data[2][4])
    # for i in range(3):
    #     plt.plot(-np.array(traj_data[i][4]), -np.array(traj_data[i][3]))
    # plt.gca().set_aspect('equal', adjustable = 'box')
    #
    # plt.show()
    # return mat_dir


if __name__ == "__main__":
    test_main()
