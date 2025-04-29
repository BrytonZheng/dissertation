from __future__ import print_function

from tqdm import tqdm
import loader2 as lo
from torch.utils.data import DataLoader
import pandas as pd
from config import *
import matplotlib.pyplot as plt
import os
import time
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

writer = pd.ExcelWriter('A.xlsx')


class Evaluate():

    def __init__(self):
        self.op = 0
        self.drawImg = True
        self.scale = 0.3048
        self.prop = 1

    def maskedMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim = 1)
        counts = t.sum(mask[:, :, 0], dim = 1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    ## Helper function for log sum exp calculation: 一个计算公式
    def logsumexp(self, inputs, dim = None, keepdim = False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim = dim, keepdim = True)
        outputs = s + (inputs - s).exp().sum(dim = dim, keepdim = True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes = 3, num_lon_classes = 2,
                      use_maneuvers = True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim = 2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim = 1)
            counts = t.sum(op_mask[:, :, 0], dim = 1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # p
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim = 1)
            counts = t.sum(op_mask[:, :, 0], dim = 1)
            return lossVal, counts, loss

    def main(self, name, val):
        model_step = 1
        # args['train_flag'] = not args['use_maneuvers']
        args['train_flag'] = not val
        l_path = args['path']
        generator = model.Generator(args = args)
        gdEncoder = model.GDEncoder(args = args)
        generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar', map_location = 'cuda:0'))
        gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.tar', map_location = 'cuda:0'))
        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()
        gdEncoder.eval()
        if val:
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset('./data/dataset_t_v_t/TestSet.mat')
                else:
                    t2 = lo.NgsimDataset('../data/5feature/TestSet.mat')
            else:
                t2 = lo.HighdDataset('Val')
            valDataloader = DataLoader(t2, batch_size = args['batch_size'], shuffle = True,
                                       num_workers = args['num_worker'],
                                       collate_fn = t2.collate_fn)  # 6716batch
        else:
            # ------------------------------------------------------------
            # a = generator.mapping
            # xx = t.tensor([[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1],
            #                [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1],
            #                [0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1]], dtype=t.float).permute(1, 0).to(
            #     device)
            # softa = t.cat(t.softmax(t.matmul(a, xx), dim=0).chunk(9, -1), dim=1).squeeze().cpu().detach().numpy()
            # a = t.cat(a.chunk(6, -1), dim=1).squeeze().cpu().detach().numpy()
            # result = np.concatenate((a, softa), axis=-1).transpose()
            # data = pd.DataFrame(result)
            # data.to_excel(writer, name, float_format='%.5f')
            # writer.save()
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset('./data/dataset_t_v_t/TestSet.mat')
                else:
                    t2 = lo.NgsimDataset('../data/5feature/TestSet.mat')
            else:
                t2 = lo.HighdDataset('Test')
            valDataloader = DataLoader(t2, batch_size = args['batch_size'], shuffle = True,
                                       num_workers = args['num_worker'],
                                       collate_fn = t2.collate_fn)

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0
        nbrsss = 0

        val_batch_count = len(valDataloader)
        print("begin.................................", name, "cnt:", val_batch_count)
        for idx, data in enumerate(tqdm(valDataloader)):
            if idx > 100:
                break

            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut[:args['out_length'], :, :]
            fut = fut.to(device)
            op_mask = op_mask[:args['out_length'], :, :]
            op_mask = op_mask.to(device)
            va = va.to(device)
            nbrsva = nbrsva.to(device)
            lane = lane.to(device)
            nbrslane = nbrslane.to(device)
            cls = cls.to(device)
            nbrscls = nbrscls.to(device)
            map_positions = map_positions.to(device)
            self.draw(hist, fut, nbrs, mask, True, lon_enc, lat_enc)

            if idx == int(val_batch_count / 4) * model_step:
                print('process:', model_step / 4)
                model_step += 1
            plt.savefig("data.png")

        # tqdm.write('valmse:', avg_val_loss / val_batch_count)
        if args['val_use_mse']:
            print('valmse:', avg_val_loss / val_batch_count)
            print(t.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
        else:
            print('valnll:', avg_val_loss / val_batch_count)
            print(lossVals / counts)
        # print(lossVals/counts*0.3048)

    def draw(self, hist, fut, nbrs, mask, train_flag, lon_man, lat_man):

        hist = hist.cpu()
        fut = fut.cpu()
        for i in range(hist.size(1)):
            print(lat_man)

            # 获取原始起点
            x_start = hist[0, i, 1] * self.scale * self.prop
            y_start = hist[0, i, 0] * self.scale

            # 计算偏移量
            x_offset = -50 - x_start
            y_offset = 0 - y_start

            # 合并数据后再进行运算
            x = torch.tensor(np.concatenate([hist[:, i, 1], fut[:, i, 1]]))
            y = torch.tensor(np.concatenate([hist[:, i, 0], fut[:, i, 0]]))

            x = x * self.scale * self.prop + x_offset
            y = y * self.scale + y_offset
            # 绘图
            plt.plot(x, y, "-", linewidth = 0.5)
            self.op += 1

            # fig.clf()


if __name__ == '__main__':

    names = ['9']
    evaluate = Evaluate()
    plt.axis('on')
    plt.ylim(-30 * evaluate.scale, 30 * evaluate.scale)
    plt.xlim(-180 * evaluate.scale * evaluate.prop, 180 * evaluate.scale * evaluate.prop)
    plt.figure(dpi = 300)

    for epoch in names:
        evaluate.main(name = epoch, val = False)
