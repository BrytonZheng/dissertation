from __future__ import print_function

import hashlib
from turtledemo.penrose import start

import torch
from torch.onnx.symbolic_opset9 import tensor

import time
import numpy as np
from tqdm import tqdm
import loader2 as lo
from torch.utils.data import DataLoader
import pandas as pd
from config import *
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
from json_pixel_to_world import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class ContinuousEvaluate():
    def __init__(self, dataset_args, output_args):
        self.op = 0
        self.draw_img = output_args["draw_img"]
        self.scale = 0.3048
        self.prop = 1
        self.draw_interval = 0
        self.draw_cur_cnt = 0
        self.dataset_dir = dataset_args['dir']
        self.ori_pic_dir = dataset_args['pic_dir']
        self.camera_params_dir = dataset_args['camera_params_dir']
        self.draw_pic = output_args['draw_pic']
        self.draw_all_pic = output_args['draw_all_pic']
        self.save_path = output_args['save_path']
        self.save_pic_path = output_args['save_pic_path']
        self.gdEncoder = None
        self.generator = None
        self.current_epoch = -1

    def main(self, name, val):
        args['train_flag'] = not val
        if self.current_epoch != int(name):
            l_path = args['path']
            self.generator = model.Generator(args = args)
            self.gdEncoder = model.GDEncoder(args = args)
            self.generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar', map_location = 'cuda:0'))
            self.gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.tar', map_location = 'cuda:0'))
            self.generator = self.generator.to(device)
            self.gdEncoder = self.gdEncoder.to(device)
            self.generator.eval()
            self.gdEncoder.eval()
        t2 = lo.NgsimDataset(self.dataset_dir, d_s = 1, t_h = 15, t_f = 25)
        valDataloader = DataLoader(t2, batch_size = args['batch_size'], num_workers = args['num_worker'],
                                   collate_fn = t2.collate_fn)

        print("epoch:", name, " cnt:", len(valDataloader))
        begin = time.time()
        pred = {}
        ok = False
        with(t.no_grad()):
            for idx, data in enumerate(tqdm(valDataloader)):
                if not ok:
                    ok = True
                    start_time = time.time()
                    print("start")
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions, dsId, vehId, frameId, refPos = data
                fut_pred = self.batch_predict(data)
                pred_to_append = fut_pred.clone()
                for i in range(hist.size(1)):
                    pred_to_append[:, i, 0] += float(refPos[i, 0].item())
                    pred_to_append[:, i, 1] += float(refPos[i, 1].item())
                    pred_to_append[:, i, [0, 1]] = -pred_to_append[:, i, [0, 1]] * self.scale
                    if vehId[i].item() not in pred:
                        pred[vehId[i].item()] = []
                    pred[vehId[i].item()].append(
                        (vehId[i].item(), frameId[i].item(), pred_to_append[:, i, :2].tolist()))
        print(time.time() - start_time, " seconds", time.time() - begin, " seconds")
        return pred

    def batch_predict(self, data):
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions, dsId, vehId, frameId, refPos = data
        hist = hist.to(device)
        nbrs = nbrs.to(device)
        mask = mask.to(device)
        lat_enc = lat_enc.to(device)
        lon_enc = lon_enc.to(device)
        fut = fut[:args['out_length'], :, :]
        fut = fut.to(device)
        op_mask = op_mask[:args['out_length'], :, :]
        op_mask = op_mask.to(device)
        va = (va / self.scale).to(device)
        nbrsva = nbrsva.to(device)
        lane = lane.to(device)
        nbrslane = nbrslane.to(device)
        cls = cls.to(device)
        nbrscls = nbrscls.to(device)
        map_positions = map_positions.to(device)

        values = self.gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
        fut_pred, lat_pred, lon_pred = self.generator(values, lat_enc, lon_enc)
        # fut_pred = self.fut_process(fut_pred, va, refPos, hist.size(1))
        if not args['train_flag']:
            indices = []
            if args['val_use_mse']:
                fut_pred_max = t.zeros_like(fut_pred[0])
                for k in range(lat_pred.shape[0]):  # 128
                    lat_man = t.argmax(lat_enc[k, :]).detach()
                    lon_man = t.argmax(lon_enc[k, :]).detach()
                    index = lon_man * 3 + lat_man
                    indices.append(index)
                    fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
            if self.draw_img:
                lat_man = t.argmax(lat_enc, dim = -1).detach()
                lon_man = t.argmax(lon_enc, dim = -1).detach()

                self.to_draw(hist, fut, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man,
                             op_mask, indices, dsId, vehId, va, frameId, refPos)
        else:
            if self.draw_img:
                lat_man = t.argmax(lat_enc, dim = -1).detach()
                lon_man = t.argmax(lon_enc, dim = -1).detach()
                self.to_draw(hist, fut, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man,
                             op_mask, None, dsId, vehId, va, frameId, refPos)
        return fut_pred

    def fut_process(self, fut_pred, va, refPos, length):
        camera_params = get_camera_params(self.camera_params_dir)
        for i in range(length):
            if va[:, i, 0][-1] < 30:
                x = float(va[:, i, 0][-1]) + 0.001
                y = x ** 0.5 * np.exp(-60 / x) + 0.05
                fut_pred = fut_pred * y

        #     # 越界判断
        #     offset_x, offset_y = float(refPos[i, 0].item()), float(refPos[i, 1].item())
        #     for fi in range(len(fut_pred[:, i, 1])):
        #         wx, wy = ((float(fut_pred[fi, i, 0].item()) + offset_x) * self.scale * self.prop,
        #                   (float(fut_pred[fi, i, 1].item()) + offset_y) * self.scale)
        #         px, py = world_ground_to_pixel(-wx, -wy, camera_params)
        #         if not (camera_params['width'] > px > 0 and
        #                 camera_params['height'] > py > camera_params['height'] / 2):
        #             fut_pred = fut_pred[:fi, :, :]
        #             break
        # print(fut_pred)
        return fut_pred

    def to_draw(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, dsId, vehId, va,
                frameId, refPos):
        if self.draw_pic:
            self.drawOriginalPNG(hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, dsId,
                                 vehId, va, frameId, refPos)
        elif self.draw_all_pic:
            self.draw_all(hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, dsId,
                          vehId, va, frameId, refPos)
        else:
            if self.draw_cur_cnt >= self.draw_interval:
                self.draw_cur_cnt = 0
            else:
                self.draw_cur_cnt += 1
                return
            self.draw(hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, dsId, vehId, va,
                      frameId, refPos)

    def draw(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, dsId, vehId, va,
             frameId, refPos):
        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        op_mask = op_mask.cpu()
        IPL = 0
        for i in range(hist.size(1)):  # 列循环
            lon_man_i = lon_man[i].item()
            lat_man_i = lat_man[i].item()

            plt.figure(dpi = 300)
            plt.autoscale(enable = False)
            plt.axis('on')
            plt.hlines(y = [-1.75, 1.75, -5.25, 5.25],
                       xmin = -300 * self.scale * self.prop,
                       xmax = 300 * self.scale * self.prop, colors = 'black', linestyles = 'dashed', linewidth = 0.3)
            plt.ylim(-18 * self.scale, 18 * self.scale)
            plt.xlim(-300 * self.scale * self.prop, 300 * self.scale * self.prop)

            # plt.figure(dpi=300, figsize=(100 * self.scale * self.prop,40 * self.scale))
            # plt.hlines([-18, -6, 6, 18], -180, 180, colors="c", linestyles="dashed")
            IPL_i = mask[i, :, :, :].sum().sum()
            IPL_i = int((IPL_i / 64).item())
            for ii in range(IPL_i):
                plt.plot(nbrs[:, IPL + ii, 1] * self.scale * self.prop, nbrs[:, IPL + ii, 0] * self.scale, ':',
                         color = 'blue',
                         linewidth = 0.5)
                # self.add_car(plt, nbrs[-1, IPL + ii, 1], nbrs[-1, IPL + ii, 0], alp=0.5)
            IPL = IPL + IPL_i
            plt.plot(hist[:, i, 1] * self.scale * self.prop, hist[:, i, 0] * self.scale, ':', color = 'red',
                     linewidth = 0.5)
            # self.add_car(plt, hist[-1, i, 1], hist[-1, i, 0], alp=1)
            plt.plot(fut[:, i, 1] * self.scale * self.prop, fut[:, i, 0] * self.scale, '-', color = 'black',
                     linewidth = 0.5)
            if train_flag:
                fut_pred = fut_pred.detach().cpu()
                plt.plot(fut_pred[:, i, 1] * self.scale * self.prop, fut_pred[:, i, 0] * self.scale, color = 'green',
                         linewidth = 0.2)
            else:
                for j in range(len(fut_pred)):
                    fut_pred_i = fut_pred[j].detach().cpu()
                    if j == indices[i].item():
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color = 'red', linewidth = 0.2)
                    else:
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color = 'green', linewidth = 0.2)
            plt.gca().set_aspect('equal', adjustable = 'box')
            save_path = os.path.join(self.save_path, str(dsId[i].item()) + '-' + str(vehId[i].item()),
                                     str(self.op) + '.png')
            print(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            plt.savefig(save_path)
            self.op += 1
            # fig.clf()
            plt.close()

    def drawOriginalPNG(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, dsId,
                        vehId, va, frameId, refPos):
        dir = self.ori_pic_dir
        camera_params = get_camera_params(self.camera_params_dir)
        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        op_mask = op_mask.cpu()
        IPL = 0
        start = 12
        for i in range(hist.size(1)):
            fid = frameId[i]
            # print("frameID:", fid)

            img = Image.open(os.path.join(dir, "Colorbox%d.png" % (start + fid * 12)))
            draw = ImageDraw.Draw(img)
            fut_pred = fut_pred.detach().cpu()
            print(fut_pred[:, i, :2].tolist(), '\n', refPos[i, :])
            world_fut_pred = []
            fut_pred[:, i, 0] += float(refPos[i, 0].item())
            fut_pred[:, i, 1] += float(refPos[i, 1].item())
            first_overflow = False
            for fi in range(len(fut_pred[:, i, 1])):
                wx, wy = float(fut_pred[fi, i, 0].item()) * self.scale * self.prop, float(
                    fut_pred[fi, i, 1].item()) * self.scale
                px, py = world_ground_to_pixel(-wx, -wy, camera_params)
                if img.size[0] > px > 0 and img.size[1] > py > 0:
                    if py < img.size[1] / 2:
                        if not first_overflow:
                            world_fut_pred.append((img.size[0] - float(px), img.size[1] - 1))
                            first_overflow = True
                    else:
                        world_fut_pred.append((float(px), float(py)))

            world_fut = []
            fut[:, i, 0] += float(refPos[i, 0].item())
            fut[:, i, 1] += float(refPos[i, 1].item())
            for fi in range(len(fut[:, i, 1])):
                wx, wy = float(fut[fi, i, 0].item()) * self.scale * self.prop, float(
                    fut[fi, i, 1].item()) * self.scale
                px, py = world_ground_to_pixel(-wx, -wy, camera_params)
                if img.size[0] > px > 0 and img.size[1] > py > img.size[1] / 2:
                    world_fut.append((float(px), float(py)))
            # particular process for future track
            while len(world_fut) >= 2 and world_fut[-2] == world_fut[-1]:
                world_fut = world_fut[:-1]
            draw.line(world_fut[:-1], fill = "yellow", width = 6)
            draw.line(world_fut_pred, fill = "red", width = 4)

            save_path = os.path.join(self.save_pic_path, str(dsId[i].item()) + '-' + str(vehId[i].item()),
                                     str(self.op) + '.png')
            print(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            img.save(save_path)
            self.op += 1

    def id_to_color(self, id):
        hash = hashlib.md5(str(id).encode()).hexdigest()
        r = int(hash[:2], 16)
        g = int(hash[2:4], 16)
        b = int(hash[4:6], 16)
        return (r, g, b)

    def get_image_size_from_ori_dir(self, ori_pic_dir):
        """读取 ori_pic_dir 下任意一张图片来确定图像尺寸"""
        for fname in os.listdir(ori_pic_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                with Image.open(os.path.join(ori_pic_dir, fname)) as img:
                    return img.size
        raise RuntimeError("No valid images found in ori_pic_dir.")

    def draw_all(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, dsId,
                 vehId, va, frameId, refPos):
        dir = self.save_pic_path
        camera_params = get_camera_params(self.camera_params_dir)

        # 获取原始图尺寸（仅一次）
        image_size = self.get_image_size_from_ori_dir(self.ori_pic_dir)

        hist = hist.cpu()
        fut_pred = fut_pred.detach().cpu()
        refPos = refPos.cpu()

        for i in range(hist.size(1)):
            fid = frameId[i].item()
            fname = f"{fid}.png"
            fpath = os.path.join(dir, fname)

            # 如果图片存在则打开，否则创建新图
            if os.path.exists(fpath):
                img = Image.open(fpath).convert("RGB")
            else:
                img = Image.new("RGB", image_size, "white")

            draw = ImageDraw.Draw(img)

            # 获取当前车辆颜色
            color = self.id_to_color(vehId[i].item())

            # 生成预测轨迹像素坐标
            traj = []
            fut_pred[:, i, 0] += float(refPos[i, 0].item())
            fut_pred[:, i, 1] += float(refPos[i, 1].item())
            for fi in range(fut_pred.size(0)):
                wx = float(fut_pred[fi, i, 0].item()) * self.scale * self.prop
                wy = float(fut_pred[fi, i, 1].item()) * self.scale
                px, py = world_ground_to_pixel(-wx, -wy, camera_params)
                if 0 < px < img.size[0] and img.size[1] / 2 < py < img.size[1]:
                    traj.append((float(px), float(py)))

            # 绘制预测轨迹
            if len(traj) > 1:
                draw.line(traj, fill = color, width = 4)

            # 保存图片
            os.makedirs(os.path.dirname(fpath), exist_ok = True)
            img.save(fpath)


if __name__ == '__main__':
    data_args, output_args = {}, {}
    data_args['dir'] = 'data/test/test/test.mat'
    data_args['pic_dir'] = 'data/test/test/output/Colorbox/'
    data_args['camera_params_dir'] = 'data/test/test/DumpSettings.json'
    # data_args['dir'] = 'roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/test.mat'
    # data_args['pic_dir'] = 'roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/output/Colorbox/'
    # data_args['camera_params_dir'] = 'roadsideCamera1-250409-111220/roadsideCamera1-250409-111220/DumpSettings.json'

    output_args['draw_img'] = True
    output_args['draw_pic'] = True
    output_args['draw_all_pic'] = False
    output_args['save_path'] = './save1/'
    output_args['save_pic_path'] = './save1_pic/'
    evaluate = ContinuousEvaluate(data_args, output_args)
    evaluate.main(name = '9', val = False)
    # evaluate.main('1', False)
