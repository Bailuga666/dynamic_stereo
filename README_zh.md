# [CVPR 2023] DynamicStereo: 从立体视频中获取一致的动态深度。

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**

[Nikita Karaev](https://nikitakaraevv.github.io/), [Ignacio Rocco](https://www.irocco.info/), [Benjamin Graham](https://ai.facebook.com/people/benjamin-graham/), [Natalia Neverova](https://nneverova.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Christian Rupprecht](https://chrirupp.github.io/)

[[`论文`](https://research.facebook.com/publications/dynamicstereo-consistent-dynamic-depth-from-stereo-videos/)] [[`项目`](https://dynamic-stereo.github.io/)] [[`BibTeX`](#citing-dynamicstereo)]

![nikita-reading](https://user-images.githubusercontent.com/37815420/236242052-e72d5605-1ab2-426c-ae8d-5c8a86d5252c.gif)

**DynamicStereo** 是一个基于变换器的架构，用于从立体视频中进行时间一致的深度估计。它在两个数据集的组合上进行了训练：[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) 和我们下面介绍的 **Dynamic Replica**。

## 数据集

https://user-images.githubusercontent.com/37815420/236239579-7877623c-716b-4074-a14e-944d095f1419.mp4

该数据集由 145200 个*立体*帧（524 个视频）组成，其中包含运动中的人和动物。

我们为*左视图和右视图*提供注释，请参见[此笔记本](https://github.com/facebookresearch/dynamic_stereo/blob/main/notebooks/Dynamic_Replica_demo.ipynb)：
- 相机内参和外参
- 图像深度（可以使用内参转换为视差）
- 实例分割掩码
- 二进制前景/背景分割掩码
- 光流（已发布！）
- 长距离像素轨迹（已发布！）

### 下载 Dynamic Replica 数据集
从[项目网站](https://dynamic-stereo.github.io/)的*数据*选项卡下载 `links.json`，接受许可协议后。
```
git clone https://github.com/facebookresearch/dynamic_stereo
cd dynamic_stereo
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```
将下载的 `links.json` 文件添加到项目文件夹。使用 `download_splits` 标志选择要下载的数据集分割：
```
python ./scripts/download_dynamic_replica.py --link_list_file links.json \
--download_folder ./dynamic_replica_data --download_splits real valid test train
```

数据集分割解压后的内存要求（包含所有注释）：
- train - 1.8T
- test - 328G
- valid - 106G
- real - 152M

您可以使用[此 PyTorch 数据集类](https://github.com/facebookresearch/dynamic_stereo/blob/dfe2907faf41b810e6bb0c146777d81cb48cb4f5/datasets/dynamic_stereo_datasets.py#L287) 来遍历数据集。

## 安装

描述使用最新 PyTorch3D、PyTorch 1.12.1 和 CUDA 11.3 安装 DynamicStereo。

### 设置所有源文件的根目录：
```
git clone https://github.com/facebookresearch/dynamic_stereo
cd dynamic_stereo
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```
### 创建 conda 环境：
```
conda create -n dynamicstereo python=3.8
conda activate dynamicstereo
```
### 安装要求
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# 安装 PyTorch3D 需要一些时间。在此期间，您可能想休息一下，喝杯咖啡。
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```

### （可选）安装 RAFT-Stereo
```
mkdir third_party
cd third_party
git clone https://github.com/princeton-vl/RAFT-Stereo
cd RAFT-Stereo
bash download_models.sh
cd ../..
```

## 评估
要下载检查点，请按照以下说明操作：
```
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_sf.pth 
wget https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_dr_sf.pth 
cd ..
```
您也可以通过点击下面的链接手动下载检查点。将检查点复制到 `./dynamic_stereo/checkpoints`。

- [DynamicStereo](https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_sf.pth) 在 SceneFlow 上训练
- [DynamicStereo](https://dl.fbaipublicfiles.com/dynamic_replica_v1/dynamic_stereo_dr_sf.pth) 在 SceneFlow 和 *Dynamic Replica* 上训练

要评估 DynamicStereo：
```
python ./evaluation/evaluate.py --config-name eval_dynamic_replica_40_frames \
 MODEL.model_name=DynamicStereoModel exp_dir=./outputs/test_dynamic_replica_ds \
 MODEL.DynamicStereoModel.model_weights=./checkpoints/dynamic_stereo_sf.pth 
```
由于图像分辨率较高，在 *Dynamic Replica* 上评估需要 32GB GPU。如果您的 GPU 内存不足，可以通过添加 `MODEL.DynamicStereoModel.kernel_size=10` 将 `kernel_size` 从 20 减少到 10。另一种选择是降低数据集分辨率。

结果，您应该看到[论文](https://arxiv.org/pdf/2305.02296.pdf)中*表 5* 的数字。（为此，您需要 `kernel_size=20`）

所有 *Dynamic Replica* 分割（包括 *real*）的重建将可视化并保存到 `exp_dir`。

如果您安装了[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)，您可以运行：
```
python ./evaluation/evaluate.py --config-name eval_dynamic_replica_40_frames \
  MODEL.model_name=RAFTStereoModel exp_dir=./outputs/test_dynamic_replica_raft
```

我们使用的其他公共数据集：
 - [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
 - [Sintel](http://sintel.is.tue.mpg.de/stereo)
 - [Middlebury](https://vision.middlebury.edu/stereo/data/)
 - [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)
 - [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)

## 训练
训练需要 32GB GPU。如果您的 GPU 内存不足，可以减少 `image_size` 和/或 `sample_len`。
您需要先下载 SceneFlow。或者，您可以只在 *Dynamic Replica* 上训练。
```
python train.py --batch_size 1 \
 --spatial_scale -0.2 0.4 --image_size 384 512 --saturation_range 0 1.4 --num_steps 200000  \
 --ckpt_path dynamicstereo_sf_dr  \
  --sample_len 5 --lr 0.0003 --train_iters 10 --valid_iters 20    \
  --num_workers 28 --save_freq 100  --update_block_3d --different_update_blocks \
  --attention_type self_stereo_temporal_update_time_update_space --train_datasets dynamic_replica things monkaa driving
```
如果您只想在 SceneFlow 上训练，请从 `train_datasets` 中移除 `dynamic_replica` 标志。

## 许可证
dynamic_stereo 的大部分内容根据 CC-BY-NC 许可证，但项目的部分内容根据单独的许可证条款提供：[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) 根据 MIT 许可证，[LoFTR](https://github.com/zju3dv/LoFTR) 和 [CREStereo](https://github.com/megvii-research/CREStereo) 根据 Apache 2.0 许可证。

## 引用 DynamicStereo
如果您在研究中使用 DynamicStereo 或 Dynamic Replica，请使用以下 BibTeX 条目。
```
@article{karaev2023dynamicstereo,
  title={DynamicStereo: Consistent Dynamic Depth from Stereo Videos},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={CVPR},
  year={2023}
}
```

步骤总结
准备图像：

将你的左右双目图像序列放在 ./my_data/left/ 和 ./my_data/right/ 文件夹中。
图像应按顺序命名，如 0001.png, 0002.png 等，确保左右对应。
运行评估：

使用预训练模型 ./checkpoints/dynamic_stereo_sf.pth（SceneFlow训练）或 ./checkpoints/dynamic_stereo_dr_sf.pth（SceneFlow + Dynamic Replica）。
查看结果：

结果保存在 ./outputs/custom_eval/visualisations/ 文件夹中。
结果形式：可视化图像（PNG格式），显示预测的视差图（disparity map），通常是彩色编码的深度/视差热图，帮助你直观看到场景的深度信息。没有数值指标，因为没有真实标签。
如果图像分辨率高，建议添加 MODEL.DynamicStereoModel.kernel_size=10 以减少内存使用。确保GPU内存足够（推荐32GB）。如果需要调整序列长度，修改 sample_len。

python prepare_data.py
python ./evaluation/evaluate.py --config-name eval_custom MODEL.DynamicStereoModel.model_weights=./checkpoints/dynamic_stereo_sf.pth
python create_depth_video.py