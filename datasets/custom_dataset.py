# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from dynamic_stereo.datasets.dynamic_stereo_datasets import StereoSequenceDataset


class CustomStereoDataset(StereoSequenceDataset):
    def __init__(self, root="./my_data", sample_len=5):
        super(CustomStereoDataset, self).__init__(aug_params=None)
        self.root = root
        self.sample_len = sample_len

        # 假设图像按顺序命名，如 left/0001.png, right/0001.png
        left_images = sorted([f for f in os.listdir(os.path.join(root, "left")) if f.endswith('.png')])
        right_images = sorted([f for f in os.listdir(os.path.join(root, "right")) if f.endswith('.png')])

        assert len(left_images) == len(right_images), "左右图像数量不匹配"

        self.image_pairs = list(zip(left_images, right_images))
        self.num_frames = len(self.image_pairs)

    def __len__(self):
        return max(1, self.num_frames - self.sample_len + 1)

    def __getitem__(self, index):
        start_idx = index
        end_idx = min(start_idx + self.sample_len, self.num_frames)

        imgs = []
        for i in range(start_idx, end_idx):
            left_path = os.path.join(self.root, "left", self.image_pairs[i][0])
            right_path = os.path.join(self.root, "right", self.image_pairs[i][1])

            left_img = Image.open(left_path).convert('RGB')
            right_img = Image.open(right_path).convert('RGB')

            # 转换为tensor
            left_tensor = torch.from_numpy(np.array(left_img)).permute(2, 0, 1).float() / 255.0
            right_tensor = torch.from_numpy(np.array(right_img)).permute(2, 0, 1).float() / 255.0

            imgs.append(torch.stack([left_tensor, right_tensor], 0))

        # 如果序列长度不够，重复最后一帧
        while len(imgs) < self.sample_len:
            imgs.append(imgs[-1])

        imgs = torch.stack(imgs, 0)  # [sample_len, 2, 3, H, W]

        return {"img": imgs}