# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from copy import deepcopy
# from models import pose_resnet, pose_resnet_dpi
from models import pose_resnet
from models.cuboid_proposal_net_soft import CuboidProposalNetSoft
from models.pose_regression_net import PoseRegressionNet
import torch.nn.functional as F
import numpy as np
import cv2

import utils.cameras as cameras
from utils.transforms import get_affine_transform, get_scale


from torch.utils.flop_counter import FlopCounterMode

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
        self.crop_bound = 'face'




    
    def _cal_distance(self, root, distance):
        if distance is None or distance == 0:
            return True
        d = torch.norm(root).item()
        return torch.norm(root).item() < distance

    def forward(
        self,
        views1=None,
        meta1=None,
        distance=None,
    ):
        all_heatmaps = []
        for view in views1:
            heatmaps = self.backbone(view)
            all_heatmaps.append(heatmaps)

        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]
        
        _, _, _, grid_centers = self.root_net(all_heatmaps, meta1)

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)





        for n in range(self.num_cand):
            index = pred[:, n, 0, 3] >= 0
            if torch.sum(index) > 0:
                # grid_center shape : (b, n, 5), (1, 10, 5)
                if self._cal_distance(grid_centers[:, n, :2], distance) == False:
                    grid_centers[:, n, 3] = 1
                    pred[:, n, :, 3] = 1
                    continue
                single_pose = self.pose_net(all_heatmaps, meta1, grid_centers[:, n])
                if min(single_pose[:,8,2], single_pose[:,14,2]) < 0 or min(single_pose[:,8,2], single_pose[:,14,2]) > 120:
                    grid_centers[:, n, 3] = -1
                    pred[:, n, :, 3] = -1
                    continue


                pred[:, n, :, 0:3] = single_pose.detach()
                del single_pose

        return pred, all_heatmaps, grid_centers


def  get_multi_person_pose_net(cfg, is_train=False, is_trt=False, engine_path=None):
    if is_trt:
        from .engine_model import EngineModel
        backbone = EngineModel(engine_path, copy=True)
    else:
        backbone = eval(cfg.BACKBONE_MODEL + ".get_pose_net")(cfg, is_train=is_train)
        print(cfg.BACKBONE_MODEL)
    model = MultiPersonPoseNetSSV(backbone, cfg)
    return model