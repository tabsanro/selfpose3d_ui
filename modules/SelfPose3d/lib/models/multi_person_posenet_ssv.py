# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from models import pose_resnet
from models.cuboid_proposal_net_soft import CuboidProposalNetSoft
from models.pose_regression_net import PoseRegressionNet
import utils.cameras as cameras

from torch.utils.flop_counter import FlopCounterMode

stream = torch.cuda.Stream()

def get_flops(model, inp):
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        model(inp)
    return flop_counter.get_total_flops()

class MultiPersonPoseNetSSV(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNetSSV, self).__init__()
        self.backbone = backbone
        self.pose_net = PoseRegressionNet(cfg)
        self.root_net = CuboidProposalNetSoft(cfg)
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

    def _cal_root_distance(self, root, distance):
        # return True if root is in lod2
        if distance is None or distance == 0:
            return True
        return torch.norm(root[:2]).item() < distance
    
    def _cal_pose_roa_distance(self, pose, roa_distance):
        # return True if pose is in roa
        if roa_distance is None or roa_distance == 0:
            return False
        center = torch.tensor([0, 0], device=pose.device)
        for i in range(15):
            if torch.norm(pose[0][i][:2] - center).item() < roa_distance:
                return True

    def forward(
        self,
        views1=None,
        meta1=None,
        distance=None,
        roa_distance=None,
        tracking_id=None,
        reid_info=None,
    ):
        all_heatmaps = []
        for view in views1:
            with torch.cuda.stream(stream):
                heatmaps = self.backbone(view)
                all_heatmaps.append(heatmaps)
        torch.cuda.current_stream().wait_stream(stream)

        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]
        
        _, _, _, grid_centers = self.root_net(all_heatmaps, meta1)

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)

        id_map_list = np.full(self.num_cand, -1)
        if reid_info:
            num_id = len(reid_info) # num id
            num_views = len(views1)
            roots_2d = np.zeros((num_views, self.num_cand, 2))
            for n in range(num_views):
                cam = {}
                for k, v in meta1[n]['camera'].items():
                    cam[k] = v[0]
                xy = cameras.project_pose(grid_centers[0, :, :3], cam)
                roots_2d[n, :] = xy.detach().cpu()

            for reid, data in reid_info.items():
                id_score = np.zeros((self.num_cand, 1))
                for d in data:
                    cam_num = d["cam"]
                    id_root = d["root_2d"] # xyxy
                    for n in range(self.num_cand):
                        l2 = np.linalg.norm(id_root - roots_2d[cam_num, n])
                        id_score[n] += l2
                id_index = np.argmin(id_score)
                id_map_list[id_index] = int(reid)
                    

        for n in range(self.num_cand):
            with torch.cuda.stream(stream):
                if id_map_list[n] == -1:
                    grid_centers[:, n, 3] = -1
                    pred[:, n, :, 3] = -1
                    continue
                # grid_center shape : (b, n, 5), (1, 10, 5)
                if self._cal_root_distance(grid_centers[:, n, :2], distance) == False and id_map_list[n] not in tracking_id:
                    grid_centers[:, n, 3] = 1
                    pred[:, n, :, 3] = 1
                    continue

                single_pose = self.pose_net(all_heatmaps, meta1, grid_centers[:, n])


                if min(single_pose[:,8,2], single_pose[:,14,2]) < - 50 or min(single_pose[:,8,2], single_pose[:,14,2]) > 250:
                    grid_centers[:, n, 3] = -1
                    pred[:, n, :, 3] = -1
                    continue

                if self._cal_pose_roa_distance(single_pose, roa_distance):
                    if id_map_list[n] not in tracking_id and id_map_list[n] != -1:
                        tracking_id.append(id_map_list[n])

                pred[:, n, :, 0:3] = single_pose.detach()
                del single_pose
        torch.cuda.current_stream().wait_stream(stream)

        return pred, all_heatmaps, grid_centers, tracking_id, id_map_list


def  get_multi_person_pose_net(cfg, is_train=False, is_trt=False, engine_path=None):
    if is_trt:
        from .engine_model import EngineModel
        backbone = EngineModel(engine_path, copy=True)
    else:
        backbone = pose_resnet.get_pose_net(cfg, is_train=is_train)
    model = MultiPersonPoseNetSSV(backbone, cfg)
    return model