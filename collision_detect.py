import continuous_evaluate as eval


class CollisionDetect():
    def __init__(self, args):
        self.model_data_args = args["data_args"]
        self.model_output_args = args["output_args"]
        self.model_epoch = str(args["epoch"])
        self.multi_model = args["multi_model"]

    def detect(self):
        pred_model = eval.ContinuousEvaluate(self.model_data_args, self.model_output_args)
        pred_model.main(name = self.model_epoch, val = self.multi_model)


if __name__ == "__main__":
    args = {
        "data_args": {
            "dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/test.mat",
            "pic_dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/output/Colorbox/",
            "camera_params_dir": "roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/DumpSettings.json"
        },
        "output_args": {
            "draw_img": True,
            "draw_pic": True,
            "save_path": "./save2/",
            "save_pic_path": "./save2_pic/",
        },
        "epoch": 9,
        "multi_model": False,
    }
    detect = CollisionDetect(args)
    detect.detect()
