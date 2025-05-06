from matplotlib import is_interactive

import continuous_evaluate as eval


class CollisionDetect():
    def __init__(self, args):
        self.model_data_args = args["data_args"]
        self.model_output_args = args["output_args"]
        self.model_epoch = str(args["epoch"])
        self.multi_model = args["multi_model"]

    def detect(self, construction_area):
        pred_model = eval.ContinuousEvaluate(self.model_data_args, self.model_output_args)
        pred = pred_model.main(name = self.model_epoch, val = self.multi_model)
        for id, v in pred.items():
            for i, pred_info in enumerate(v):
                traj = pred_info[2]
                collide_idx = self.collision(construction_area, traj)
                if collide_idx != len(traj):
                    print(f"{i} Collision {id} detected at index {collide_idx}")
                    continue

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


if __name__ == "__main__":
    args = {
        "data_args": {
            "dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/test.mat",
            "pic_dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/output/Colorbox/",
            "camera_params_dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/DumpSettings.json"
        },
        "output_args": {
            "draw_img": False,
            "draw_pic": False,
            "save_path": "./save2/",
            "save_pic_path": "./save2_pic/",
        },
        "epoch": 9,
        "multi_model": False,
    }
    detect = CollisionDetect(args)
    detect.detect([(-93.69368367231141, 196.1469398680318), (-99.0701578236929, 196.30828249826672),
                   (-98.24634426018073, 223.1893222104566), (-94.44939174710971, 290.64717150073693)])
    # detect.detect([(-94.44939174710971, 290.64717150073693), (-98.24634426018073, 223.1893222104566),
    #                (-99.0701578236929, 196.30828249826672), (-93.69368367231141, 196.1469398680318)])
