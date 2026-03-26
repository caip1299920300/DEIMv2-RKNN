# 1、环境搭建
```
# 创建指定Python版本的环境
uv venv --python 3.11
# 或者直接指定Python解释器路径（如果3.11已安装）
uv venv --python python3.11
source .venv/bin/activate
uv pip install rknn-toolkit2 -i https://pypi.mirrors.ustc.edu.cn/simple/
uv pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

# 2、下载pth模型
| Model | Dataset | AP | #Params | GFLOPs | Latency (ms) | config | Hugging Face | checkpoint | log |
| :---: | :---: | :---: | :---: | :---: |:------------:| :---: | :---: | :---: | :---: |
| **Atto** | COCO | **23.8** | 0.5M | 0.8 | 1.10 | [yml](./configs/deimv2/deimv2_hgnetv2_atto_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_ATTO_COCO) | [Google](https://drive.google.com/file/d/18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc/view?usp=sharing) / [Quark](https://pan.quark.cn/s/04c997582fca) | [Google](https://drive.google.com/file/d/1M7FLN8EeVHG02kegPN-Wxf_9BlkghZfj/view?usp=sharing) / [Quark](https://pan.quark.cn/s/7bf3548d3e10) |
| **Femto** | COCO | **31.0** | 1.0M | 1.7 | 1.45 | [yml](./configs/deimv2/deimv2_hgnetv2_femto_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_FEMTO_COCO) | [Google](https://drive.google.com/file/d/16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds/view?usp=sharing) / [Quark](https://pan.quark.cn/s/169f3cefec1b) | [Google](https://drive.google.com/file/d/1_KWVfOr3bB5TMHTNOmDIAO-tZJmKB9-b/view?usp=sharing) / [Quark](https://pan.quark.cn/s/9dd5c4940199) |
| **Pico** | COCO | **38.5** | 1.5M | 5.2 | 2.13 | [yml](./configs/deimv2/deimv2_hgnetv2_pico_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO) | [Google](https://drive.google.com/file/d/1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT/view?usp=sharing) / [Quark](https://pan.quark.cn/s/0db5b1dff721) | [Google](https://drive.google.com/file/d/1GwyWotYSKmFQdVN9k2MM6atogpbh0lo1/view?usp=sharing) / [Quark](https://pan.quark.cn/s/5ab2a74bb867) |
| **N** | COCO | **43.0** | 3.6M | 6.8 | 2.32 | [yml](./configs/deimv2/deimv2_hgnetv2_n_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_N_COCO) | [Google](https://drive.google.com/file/d/1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K/view?usp=sharing) / [Quark](https://pan.quark.cn/s/1f626f191d11) | [Google](https://drive.google.com/file/d/1QhYfRrUy8HrihD3OwOMJLC-ATr97GInV/view?usp=sharing) / [Quark](https://pan.quark.cn/s/54e5c89675b3) |
| **S** | COCO | **50.9** | 9.7M | 25.6 | 5.78 | [yml](./configs/deimv2/deimv2_dinov3_s_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO) | [Google](https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing) / [Quark](https://pan.quark.cn/s/f4d05c349a24) | [Google](https://drive.google.com/file/d/1ydA4lWiTYusV1s3WHq5jSxIq39oxy-Nf/view?usp=sharing) / [Quark](https://pan.quark.cn/s/277660d785d2) |
| **M** | COCO | **53.0** | 18.1M | 52.2 | 8.80 | [yml](./configs/deimv2/deimv2_dinov3_m_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_DINOv3_M_COCO) | [Google](https://drive.google.com/file/d/1nPKDHrotusQ748O1cQXJfi5wdShq6bKp/view?usp=sharing) / [Quark](https://pan.quark.cn/s/68a719248756) | [Google](https://drive.google.com/file/d/1i05Q1-O9UH-2Vb52FpFJ4mBG523GUqJU/view?usp=sharing) / [Quark](https://pan.quark.cn/s/32af04f3e4b4) |
| **L** | COCO | **56.0** | 32.2M | 96.7 | 10.47 | [yml](./configs/deimv2/deimv2_dinov3_l_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO) | [Google](https://drive.google.com/file/d/1dRJfVHr9HtpdvaHlnQP460yPVHynMray/view?usp=sharing) / [Quark](https://pan.quark.cn/s/966b7ef89bdf) | [Google](https://drive.google.com/file/d/13mrQxyrf1kJ45Yd692UQwdb7lpGoqsiS/view?usp=sharing) / [Quark](https://pan.quark.cn/s/182bd52562a7) |
| **X** | COCO | **57.8** | 50.3M | 151.6 | 13.75 | [yml](./configs/deimv2/deimv2_dinov3_x_coco.yml) | [huggingface](https://huggingface.co/Intellindust/DEIMv2_DINOv3_X_COCO) | [Google](https://drive.google.com/file/d/1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7/view?usp=sharing) / [Quark](https://pan.quark.cn/s/038aa966b283) | [Google](https://drive.google.com/file/d/13QV0SwJw1wHl0xHWflZj1KstBUAovSsV/view?usp=drive_link) / [Quark](https://pan.quark.cn/s/333aba42b4bb) |

# 3、导出onnx模型
```
# 注意事项
（1）使用Base models时。因为rknn不支持GridSample双线性插值算子，所以得切换到discrete最近邻差值。
configs/base/rtdetrv2_r50vd.yml目录中涉及到 RTDETRTransformerv2的cross_attn_method 改成 discrete，后续rknn使
（2）修改了engine/backbone/dinov3/layers/rope_position_encoding.py的RopePositionEmbedding-forward函数，否知导出onnx会有冗余的算子
uv run export_onnx.py --config ../../configs/deimv2/deimv2_dinov3_l_coco.yml  --resume=./deimv2_dinov3_l_coco.pth --simplify
```

# 4、onnx模型推理
```
uv run onnx_infer.py
```

# 4、导出rknn模型
```
uv run export_rknn.py deimv2_dinov3_l_coco.onnx rk3588
```
