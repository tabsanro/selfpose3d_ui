import torch
from torch2trt import torch2trt
import os

def export_tensorrt(model, ckpt, cfg):
    output_dir = os.path.join(ckpt.split(".")[0], 'engine')
    if os.path.isdir(output_dir):
        return
    
    print("Exporting Model to TensorRT engine...")


    backbone = model.module.backbone
    root_v2v_net = model.module.root_net.v2v_net
    pose_v2v_net = model.module.pose_net.v2v_net

    backbone_x = torch.ones((1, 3, cfg.NETWORK.IMAGE_SIZE[1], cfg.NETWORK.IMAGE_SIZE[0])).cuda()
    root_v2v_net_x = torch.ones((1, 1, cfg.MULTI_PERSON.INITIAL_CUBE_SIZE[0], cfg.MULTI_PERSON.INITIAL_CUBE_SIZE[1], cfg.MULTI_PERSON.INITIAL_CUBE_SIZE[2])).cuda()
    pose_v2v_net_x = torch.ones((1, cfg.NETWORK.NUM_JOINTS, cfg.PICT_STRUCT.CUBE_SIZE[0], cfg.PICT_STRUCT.CUBE_SIZE[1], cfg.PICT_STRUCT.CUBE_SIZE[2])).cuda()
    
    backbone_trt = torch2trt(backbone, [backbone_x], fp16_mode=True, use_onnx=True)
    root_v2v_net_trt = torch2trt(root_v2v_net, [root_v2v_net_x], fp16_mode=True, use_onnx=True)
    pose_v2v_net_trt = torch2trt(pose_v2v_net, [pose_v2v_net_x], fp16_mode=True, use_onnx=True)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(backbone_trt.state_dict(), os.path.join(output_dir, 'backbone.pth'))
    torch.save(root_v2v_net_trt.state_dict(), os.path.join(output_dir, 'root_v2v_net.pth'))
    torch.save(pose_v2v_net_trt.state_dict(), os.path.join(output_dir, 'pose_v2v_net.pth'))
    print("Exporting Model to TensorRT engine... Done")
    