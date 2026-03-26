"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            self.orig_target_sizes =  torch.tensor([1920,1080])

        def forward(self, images):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, self.orig_target_sizes)
            return outputs

    model = Model()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    data = torch.rand(1, 3, *img_size)
    _ = model(data)

    dynamic_axes = {
        # 'images': {0: 'N', }
    }

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    torch.onnx.export(
        model,
        data,
        output_file,
        input_names=['images'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        import numpy as np
        
        dynamic = True  # 或者根据您的需求设置
        
        # 修复：当 dynamic=True 时，不应该使用固定的 input_shapes
        # 应该使用 None 或者只指定动态维度
        if dynamic:
            # 方式1：不指定测试输入形状，让 onnxsim 自动处理
            input_shapes = None
            # 方式2：或者指定动态维度（推荐）
            # input_shapes = {'images': [1, 3, -1, -1]}  # -1 表示动态维度
        else:
            input_shapes = {'images': data.shape}
        
        try:
            # 添加更多简化选项
            onnx_model_simplify, check = onnxsim.simplify(
                output_file,
                test_input_shapes=input_shapes,
                # 添加以下参数以处理动态模型
                dynamic_input_shape=dynamic,  # 关键参数
                # 如果仍然失败，可以尝试禁用某些优化
                # skip_fuse_bn=False,
                # skip_constant_folding=False,
            )
            
            if check:
                onnx.save(onnx_model_simplify, output_file)
                print(f'Successfully simplified onnx model')
            else:
                print(f'Simplification check failed, but model might still be usable')
                onnx.save(onnx_model_simplify, output_file + '.simplified')
                print(f'Saved simplified model as {output_file}.simplified')
                
        except Exception as e:
            print(f'Simplification failed: {e}')
            print('Using original ONNX model without simplification')
            # 保留原始模型，继续执行


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--opset', type=int, default=17,)
    parser.add_argument('--check',  action='store_true')
    parser.add_argument('--simplify',  action='store_true')
    args = parser.parse_args()
    main(args)
