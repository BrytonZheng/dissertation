from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py
from config import args
import time


class NgsimDataset(Dataset):

    def __init__(self, mat_file, t_h = 30, t_f = 50, d_s = 2, enc_size = 64,
                 grid_size = (13, 3), window_step = 2):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # 
        self.t_f = t_f  # 
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid
        self.alltime = 0
        self.count = 0
        self.window_step = window_step  # for continuous loading by slide window
        self.window_len = self.t_h + self.t_f  # size of slide window

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return idx, self.getItem(idx)

    def getItem(self, idx, delta_t = 0):
        """
            getItem is used for getting a car track and neighbors' information from preprocessed mat file.

        Parameters:
            idx: index of main car track you want to get.
            delta_t: start frame offset of return values. default = 0
        """

        dsId = self.D[idx, 0].astype(int)  # dataset id
        vehId = self.D[idx, 1].astype(int)  # vehicle id
        t = self.D[idx, 2] + delta_t  # frame
        grid = self.D[idx, 11:]  # grid id
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([args['lon_length']])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([args['lat_length']])
        lat_enc[int(self.D[idx, 9] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
        return (hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                neighborsdistance, cclass, neighborsclass, dsId, vehId, t, refPos)

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        """
            Retrieve the historical velocity and acceleration of a vehicle.

        Parameters:
            vehId: ID of target vehicle.
            t: frame at which prediction starts.
            refVehId: vehicle ID used as velocity reference frame. TODO: 其实没用
            dsId: dataset ID.

        Returns:
            np.ndarray: A 2D array of shape (N, 2), where each row (v, a) represents
                        the velocity and acceleration at a specific timestep in history.
        """
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                # get history location
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        """
           Retrieve the historical location of a vehicle.

        Parameters:
            vehId: ID of target vehicle.
            t: frame at which prediction starts.
            refVehId: vehicle ID used as location reference frame.
            dsId: dataset ID.

        Returns:
            np.ndarray: A 2D array of shape (N, 2), where each row (x, y) represents
                        the local_X and local_Y at a specific timestep in history.
        """
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        """
            Retrieve the historical location distance between target vehicle and reference vehicle.

        Parameters:
            vehId: ID of target vehicle.
            t: frame at which prediction starts.
            refVehId: vehicle ID used as location reference frame.
            dsId: dataset ID.

        Returns:
            np.ndarray: An array of length N, where each item represents the distance
                        between target vehicle and reference vehicle at a specific timestep in history.
        """
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        """
            Retrieve future track of a vehicle.
            This function is guaranteed to return a track of length t_f.
        """
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):
        """
            collate function used for torch.Dataloader.

        Returns:
            Arrays of shape (N, batch_size, dim).
        """
        ttt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, (_, _, nbrs, *_) in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), args['lat_length'])  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), args['lon_length'])  # (batch,2)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        dsId_batch = torch.zeros(len(samples), dtype = torch.int)
        vehId_batch = torch.zeros(len(samples), dtype = torch.int)
        frameId_batch = torch.zeros(len(samples), dtype = torch.int)
        refPos_batch = torch.zeros(len(samples), 2)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (idx, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                             neighborsdistance, cclass, neighborsclass, dsId, vehId, frameId, refPos)) in enumerate(
            samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            dsId_batch[sampleId] = dsId
            vehId_batch[sampleId] = vehId
            frameId_batch[sampleId] = frameId
            refPos_batch[sampleId, 0] = refPos[0]
            refPos_batch[sampleId, 1] = refPos[1]

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        #  mask_batch 
        self.alltime += (time.time() - ttt)
        self.count += args['num_worker']
        # if (self.count > args['time']):
        #    print(self.alltime / self.count, "data load time")
        return (hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch,
                va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch,
                nbrsclass_batch, map_position, dsId_batch, vehId_batch, frameId_batch, refPos_batch)


class HighdDataset(Dataset):

    def __init__(self, mat_file, t_h = 30, t_f = 50, d_s = 2, enc_size = 64,
                 grid_size = (13, 3)):
        self.D = np.transpose(h5py.File(mat_file, 'r')['traj'].value)
        self.T = h5py.File(mat_file, 'r')
        ref = self.T['tracks'][0][0]
        res = self.T[ref]
        self.t_h = t_h  # 
        self.t_f = t_f  # 
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[0, idx].astype(int)
        vehId = self.D[1, idx].astype(int)
        t = self.D[2, idx]
        grid = self.D[14:, idx]
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(
                self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(
                self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 13] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 12] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        return hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 8]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 8]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6:8]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6:8] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(
            vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):
        nowt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, nbrs, *_ in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)  # (len,batch*车数，2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],
                                 self.grid_size[0],
                                 self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples),
                                2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples),
                                    2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), 2)  # (batch,2)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (
                hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane,
                neighborslane,
                refdistance,
                neighborsdistance, cclass, neighborsclass) in enumerate(
            samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(
                refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(
                        nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(
                        nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(
                        self.enc_size).byte()
                    map_position = torch.cat(
                        (map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(
                        nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(
                        nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2,
                    :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(
                        nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4,
                    :] = torch.from_numpy(nbrclass)
                    count4 += 1
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position
