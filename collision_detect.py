import glob
import os

from PIL import Image, ImageDraw
from PIL.ImageFont import ImageFont, load_default, truetype

import continuous_evaluate as eval
from json_pixel_to_world import pixel_to_world_ground, get_camera_params, world_ground_to_pixel


class CollisionDetect():
    def __init__(self, args):
        self.model_data_args = args["data_args"]
        self.model_output_args = args["output_args"]
        self.model_epoch = str(args["epoch"])
        self.multi_model = args["multi_model"]
        self.current_epoch = args["epoch"]
        self.camera_params = None
        self.pred_model = eval.ContinuousEvaluate(self.model_data_args, self.model_output_args)
        self.image_cnt = 0
        if self.model_output_args['draw_img']:
            if self.model_output_args['draw_all_pic']:
                self.image_size = self.pred_model.get_image_size_from_ori_dir()

    def detect(self, construction_area):
        pred = self.pred_model.main(name = self.model_epoch, val = self.multi_model)

        # convert construction_area coordinate unit
        self.camera_params = get_camera_params(self.model_data_args["camera_params_dir"])
        construction_area = self.convert_to_pixel(construction_area)
        print("construction_area: ", construction_area)

        collision_indexes = []
        for id, v in pred.items():
            for pred_info in v:
                traj = pred_info[2].copy()
                collide_idx = self.collision(construction_area, traj)
                if self.model_output_args['draw_img']:
                    if self.model_output_args['draw_pic'] and self.model_output_args['observed_car'] == id:
                        self.draw_collision_img(pred_info[1], id, traj, collide_idx, pred_info[3].copy())
                    elif self.model_output_args['draw_all_pic']:
                        self.draw_all_collision_img(pred_info[1], id, traj, collide_idx)
                if collide_idx != len(traj):
                    print(f"frameID {pred_info[1]}: Collision vid {id} detected at prediction index {collide_idx}")
                    collision_indexes.append(collide_idx)
                    continue
        if self.model_output_args['draw_img'] and self.model_output_args['draw_all_pic']:
            self.merge_images()

    def convert_to_pixel(self, points):
        ret = []
        for (px, py) in points:
            ret.append(pixel_to_world_ground(px, py, self.camera_params))
        return ret

    def collision(self, construction_area, predicted_trajectory):
        """
        Check if the predicted trajectory collides with the construction area.

        Args:
            construction_area (list): List of counter-clockwise ordered points defining the construction area.
            predicted_trajectory (list): List of points defining the predicted trajectory.
        Returns:
            The index of the nearest point to the current time in the first detected collision trajectory segment.
            If no collision is detected, return the length of the predicted trajectory.
        """
        predicted_trajectory.append(predicted_trajectory[0])
        # 判断交集
        for i in range(len(predicted_trajectory) - 1):
            p_pred1, p_pred2 = predicted_trajectory[i], predicted_trajectory[i + 1]
            for j in range(len(construction_area) - 1):
                p_con1, p_con2 = construction_area[j], construction_area[j + 1]
                if self.is_intersect(p_pred1, p_pred2, p_con1, p_con2):
                    return i
        return len(predicted_trajectory)

    def is_intersect(self, p1, p2, p3, p4):
        """
        Check if two line segments intersect.

        Args:
            p1, p2: Points defining the first line segment.
            p3, p4: Points defining the second line segment.

        Returns:
            True if the line segments intersect, False otherwise.
        """

        def ccw(A, B, C):
            return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

        # 四方向叉积
        ccw1 = ccw(p1, p2, p3)  # AB -> AC
        ccw2 = ccw(p1, p2, p4)  # AB -> AD
        ccw3 = ccw(p3, p4, p1)  # CD -> CA
        ccw4 = ccw(p3, p4, p2)  # CD -> CB

        # 一般情况
        if (ccw1 * ccw2 < 0) and (ccw3 * ccw4 < 0):
            return True

        # C是否在AB线段上
        def on_segment(A, B, C):
            return (min(A[0], B[0]) <= C[0] <= max(A[0], B[0]) and
                    min(A[1], B[1]) <= C[1] <= max(A[1], B[1]))

        if ccw1 == 0 and on_segment(p1, p2, p3):
            return True
        if ccw2 == 0 and on_segment(p1, p2, p4):
            return True
        if ccw3 == 0 and on_segment(p3, p4, p1):
            return True
        if ccw4 == 0 and on_segment(p3, p4, p2):
            return True

        return False

    def draw_collision_img(self, frame_id, veh_id, fut_pred, collide_index, fut):
        save_path = self.model_output_args['save_pic_path']
        camera_params = self.camera_params
        start = 24
        img = Image.open(os.path.join(self.pred_model.ori_pic_dir, "Colorbox%d.png" % (start + frame_id * 12)))
        draw = ImageDraw.Draw(img)

        # 预测路径
        normal_pred = []
        collide_pred = []
        for i, pt in enumerate(fut_pred):
            wx, wy = pt[0], pt[1]
            px, py = world_ground_to_pixel(wx, wy, camera_params)
            if i == 0:
                pos = (px, py)
            if img.size[0] > px > 0 and img.size[1] > py > img.size[1] / 2:
                if i < collide_index:
                    normal_pred.append((float(px), float(py)))
                else:
                    collide_pred.append((float(px), float(py)))

        while len(collide_pred) > 0 and (
                (len(normal_pred) > 0 and collide_pred[-1] == normal_pred[0])
                or
                collide_pred[-1] == collide_pred[0]
        ):
            collide_pred = collide_pred[:-1]
        while len(normal_pred) > 0 and normal_pred[-1] == normal_pred[0]:
            normal_pred = normal_pred[:-1]
        if len(collide_pred) > 0:
            normal_pred.append(collide_pred[0])

        # 实际路径
        future = []
        for i, pt in enumerate(fut):
            wx, wy = pt[0], pt[1]
            px, py = world_ground_to_pixel(wx, wy, camera_params)
            if img.size[0] > px > 0 and img.size[1] > py > img.size[1] / 2:
                future.append((float(px), float(py)))
        while len(future) > 1 and future[-1] == future[-2]:
            future = future[:-1]

        draw.line(normal_pred, fill = "red", width = 4)
        draw.line(collide_pred, fill = "blue", width = 4)
        draw.line(future[:-1], fill = "yellow", width = 4)
        collide_text, fill_color, font = self.get_collide_font(pos, collide_index)
        draw.text(pos, str(veh_id) + collide_text, fill = fill_color, font = font)

        save_path = os.path.join(save_path, '1-' + str(veh_id), str(self.image_cnt) + '.png')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        img.save(save_path)
        self.image_cnt += 1

    def draw_all_collision_img(self, frame_id, veh_id, fut_pred, collide_index):
        save_path = self.model_output_args['save_pic_path']
        camera_params = self.camera_params
        image_size = self.image_size
        fname = f"{frame_id}.png"
        fpath = os.path.join(save_path, fname)

        if os.path.exists(fpath):
            img = Image.open(fpath).convert("RGBA")
        else:
            img = Image.new("RGBA", image_size, (255, 255, 255, 0))

        draw = ImageDraw.Draw(img)
        color1, color2 = self.pred_model.id_to_color(veh_id)

        normal_pred = []
        collide_pred = []
        for i, pt in enumerate(fut_pred):
            wx, wy = pt[0], pt[1]
            px, py = world_ground_to_pixel(wx, wy, camera_params)
            if i == 0:
                pos = (px, py)
            if img.size[0] > px > 0 and img.size[1] > py > img.size[1] / 2:
                if i < collide_index:
                    normal_pred.append((float(px), float(py)))
                else:
                    collide_pred.append((float(px), float(py)))

        while len(collide_pred) > 0 and (
                (len(normal_pred) > 0 and collide_pred[-1] == normal_pred[0])
                or
                collide_pred[-1] == collide_pred[0]
        ):
            collide_pred = collide_pred[:-1]
        while len(normal_pred) > 0 and normal_pred[-1] == normal_pred[0]:
            normal_pred = normal_pred[:-1]
        if len(collide_pred) > 0:
            normal_pred.append(collide_pred[0])

        draw.line(normal_pred, fill = color1, width = 4)
        draw.line(collide_pred, fill = color2, width = 4)

        collide_text, fill_color, font = self.get_collide_font(pos, collide_index)
        draw.text(pos, str(veh_id) + collide_text, fill = fill_color, font = font)
        os.makedirs(os.path.dirname(fpath), exist_ok = True)
        img.save(fpath)

    def get_collide_font(self, pos, collide_index):
        collide_text = "OK"
        fill_color = (0, 255, 0)
        if 10 <= collide_index <= 20:
            collide_text = "Warning"
            fill_color = (255, 255, 0)
        elif collide_index < 10:
            collide_text = "Danger"
            fill_color = (255, 0, 0)

        min_font_size = 20
        max_font_size = 200
        y_ratio = (pos[1] - self.camera_params["height"] / 2) / (self.camera_params["height"] / 2)
        font_size = int(min_font_size + y_ratio * (max_font_size - min_font_size))
        font_path = "C:/Windows/Fonts/simsun.ttc"
        if os.path.exists(font_path):
            font = truetype(font_path, size = font_size)
        else:
            font = load_default(size = font_size)
        return collide_text, fill_color, font

    def merge_images(self):
        pic_path = self.pred_model.ori_pic_dir
        save_path = self.model_output_args['save_pic_path']
        for traj_file in glob.glob(os.path.join(save_path, "*.png")):
            traj_name = os.path.basename(traj_file)
            frame_id_str, _ = os.path.splitext(traj_name)
            if not frame_id_str.isdigit():
                continue
            frame_id = int(frame_id_str)

            pic_id = 24 + frame_id * 12
            pic_name = f"Colorbox{pic_id}.png"
            pic_file = os.path.join(pic_path, pic_name)

            if not os.path.exists(pic_file):
                print(f"未找到对应原图: {pic_file}")
                continue

            # 打开两张图
            pic_img = Image.open(pic_file).convert("RGBA")
            traj_img = Image.open(traj_file).convert("RGBA")

            if traj_img.size != pic_img.size:
                traj_img = traj_img.resize(pic_img.size)

            # 图像叠加
            merged = pic_img.copy()
            merged.paste(traj_img, (0, 0), traj_img)

            # 保存合成图
            output_file = os.path.join(save_path, traj_name)
            merged.save(output_file)
        # img = Image.open(os.path.join(self.pred_model.ori_pic_dir, "Colorbox%d.png" % (start + frame_id * 12)))


if __name__ == "__main__":
    args = {
        "data_args": {
            # "dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/test.mat",
            # "pic_dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/output/Colorbox/",
            # "camera_params_dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/DumpSettings.json"
            "dir": 'data/test/test/test.mat',
            "pic_dir": 'data/test/test/output/Colorbox/',
            "camera_params_dir": 'data/test/test/DumpSettings.json',
        },
        "output_args": {
            "draw_img": True,
            "draw_pic": False,
            "draw_all_pic": False,
            "save_path": "./save1/",
            "save_pic_path": "./save1_pic/",
            "observed_car": 2,
        },
        "epoch": 9,
        "multi_model": False,
    }
    detect = CollisionDetect(args)
    # detect.detect([(-93.69368367231141, 196.1469398680318), (-99.0701578236929, 196.30828249826672),
    #                (-98.24634426018073, 223.1893222104566), (-94.44939174710971, 290.64717150073693)])
    # detect.detect([(-94.44939174710971, 290.64717150073693), (-98.24634426018073, 223.1893222104566),
    #                (-99.0701578236929, 196.30828249826672), (-93.69368367231141, 196.1469398680318)])
    detect.detect([(1400, 1200), (1140, 1200), (1000, 1550), (1700, 1500)])
