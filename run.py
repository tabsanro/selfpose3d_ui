import os
import sys
import argparse
from typing import List, Tuple, Union

import torch
from tqdm import tqdm
from PyQt5 import QtWidgets, QtCore

from configs.config import config as focus_config
from configs.config import update_config as update_focus_config
from modules.focus.dataset import FOCUSDataset as dataset
from modules.focus.tensorrt import export_tensorrt, load_tensorrt_model
from modules.SelfPose3d.lib.models.multi_person_posenet_ssv import get_multi_person_pose_net
from modules.SelfPose3d.lib.core.config import config as sp3d_config
from modules.SelfPose3d.lib.core.config import update_config as update_sp3d_config
from modules.realsense.realsense import set_pipelines
from modules.gui.plot_widget import PlotWidget, DISTANCE

CWD = os.getcwd()

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch AISL Inference")
    parser.add_argument("--cfg_focus", default='configs/focus.yaml', help="experiment configure file name", type=str)
    # parser.add_argument("--cfg_sp3d", default='modules/SelfPose3d/config/cam4_posenet.yaml', help="experiment configure file name", type=str)
    parser.add_argument("--source_folder", default=None, help="source folder name", type=str)
    parser.add_argument("--tensorrt", action="store_true", default=False, help="If set, the program will use tensorrt.")
    parser.add_argument("--webcam", action="store_true", default=False, help="If set, the program will use webcam.")
    args, rest = parser.parse_known_args()
    if args.webcam:
        parser.add_argument("--webcam_info", type=str, default=None, help="Webcam info file path.")
    return parser

# Factory method
def get_sources_and_calibs(cfg, args) -> Union[Tuple[List[str], str], Tuple[None, None]]:
    if args.webcam:
        return None, None
    elif args.source_folder is None:
        sources = [os.path.join('modules', 'SelfPose3d', 'data_0705', 'hdVideos', 'hd_00_{:02d}.mp4'.format(i+1)) for i in range(cfg.NUM_SOURCES)]
        calib_path = os.path.join('modules', 'SelfPose3d', 'data_0705', 'calibration')
        return sources, calib_path
    else:
        sources = [os.path.join(args.source_folder, 'hdVideos', 'hd_00_{:02d}.mp4'.format(i+1)) for i in range(cfg.NUM_SOURCES)]
        calib_path = os.path.join(args.source_folder, 'calibration')
        return sources, calib_path

def post_process(preds_3d, grid_centers):
    # POSE
    # -1 is not person
    # 0 is in lod2
    # 1 is in lod1
    # we predict always total 10 persons
    lod_list = []
    preds_3d = preds_3d[preds_3d[...,3]!=-1]  
    grid_centers = grid_centers[grid_centers[...,3]!=-1]
    # in_lod = preds_3d[preds_3d[...,3]==0].view(-1, 15, 5)
    in_lod = grid_centers[grid_centers[...,3]==0].view(-1, 5)
    num_person = in_lod.shape[0]
    for value in grid_centers[...,3]:
        if value.item() == 0:
            lod_list.append(2)
        else:
            lod_list.append(1)
    
    if preds_3d.size(0) == 0:
        preds_3d = None # No person detected in lod2
    else:
        preds_3d = preds_3d.squeeze(0).view(-1, 15, 5).detach().cpu().numpy()  # (batch, n, 15, 5)
    # No person detected in lod1
    if grid_centers.size(0) == 0:
        grid_centers = None
    else:
        grid_centers = grid_centers.squeeze(0).view(-1, 5).detach().cpu().numpy()
    
    return preds_3d, grid_centers, num_person, lod_list

@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    # Update config
    update_focus_config(args.cfg_focus)
    update_sp3d_config(focus_config.CONFIG.POSENET)

    sources, calib_path = get_sources_and_calibs(focus_config, args)
    pipelines = None
    if args.webcam:
        pipelines = set_pipelines(args.webcam_info)
    
    # Set CUDA device
    gpus = [0]
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = sp3d_config.CUDNN.ENABLED
        torch.backends.cudnn.benchmark = sp3d_config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = sp3d_config.CUDNN.DETERMINISTIC
        device = torch.device('cuda')
    else:
        raise ValueError('CUDA is not available. Please check your configuration.')

    data_set = dataset(focus_config, sources=sources, calib_path=calib_path, pipelines=pipelines, device=device)
    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False,
    )

    # set model
    temp_model = get_multi_person_pose_net(
        sp3d_config,
        is_train=False,
    )
    temp_model = torch.nn.DataParallel(temp_model, device_ids=[0]).cuda()
    temp_model.module.load_state_dict(torch.load(focus_config.MODEL.POSENET.CKPT, weights_only=False))
    temp_model = temp_model.eval()

    if args.tensorrt:
        tensorrt_dir = os.path.join(focus_config.MODEL.POSENET.CKPT.split(".")[0], 'engine')
        if not os.path.isdir(tensorrt_dir):
            export_tensorrt(temp_model, tensorrt_dir, sp3d_config)
        pose_model = load_tensorrt_model(temp_model, tensorrt_dir)
    else:
        pose_model = temp_model

    # visualize
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = PlotWidget()
    mainWindow.show()

    # Inference
    for origin_frames, transed_frames, meta in (pbar := tqdm(data_loader)):
        # Update distance
        distance = mainWindow.pose_updater.distance
        
        # Set Results
        results = [] # 사람 수 만큼 결과 저장
        # 사람 = {lod , root, pred, age, gender}

        # Pose Estimation
        pred_3d, _, roots = pose_model(
            views1=transed_frames,
            meta1=meta,
            distance=distance,
        )
        pred_3d, roots, num_person, lod_list = post_process(pred_3d, roots)
        if roots is None:
            continue
        for num_roots in range(len(roots)):
        # for lod, pred, root in zip(lod_list, pred_3d, roots):
            temp_dict = {}
            temp_dict['lod'] = lod_list[num_roots]
            temp_dict['root'] = roots[num_roots]
            if pred_3d is not None:
                # WARNING: lod2인데 pose가 없는 경우도 있음
                # 어차피 영상이다보니 이런 경우는 출력을 하지 않아도 시계열 연속성이 있는 것 처럼 보임
                # 다음 프레임에서 pose를 딸 확률이 높기 때문에
                try:
                    temp_dict['pred'] = pred_3d[num_roots]
                except:
                    temp_dict['pred'] = None
            else:
                temp_dict['pred'] = None
            results.append(temp_dict)

        mainWindow.pose_updater.update_pose(results)

        QtCore.QCoreApplication.processEvents()
    sys.exit(app.exec_())

if __name__ == '__main__':
    default_argv=[
        '--cfg_focus', 'configs/focus.yaml',
        '--source_folder', 'modules/SelfPose3d/data_0325',
        '--tensorrt',
    ]
    # CLI 인자가 없을 때 기본 argv를 사용하도록 함
    if len(sys.argv) == 1:
        sys.argv.extend(default_argv)
    main()
