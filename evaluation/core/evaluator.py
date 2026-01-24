# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import Configurable

from dynamic_stereo.evaluation.utils.eval_utils import depth2disparity_scale, eval_batch
from dynamic_stereo.evaluation.utils.utils import (
    PerceptionPrediction,
    pretty_print_perception_metrics,
    visualize_batch,
)


class Evaluator(Configurable):
    """
    A class defining the DynamicStereo evaluator.

    Args:
        eps: Threshold for converting disparity to depth.
    """

    eps = 1e-5

    def setup_visualization(self, cfg: DictConfig) -> None:
        # Visualization
        self.visualize_interval = cfg.visualize_interval
        self.exp_dir = cfg.exp_dir
        if self.visualize_interval > 0:
            self.visualize_dir = os.path.join(cfg.exp_dir, "visualisations")

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        is_real_data: bool = False,
        step=None,
        writer=None,
        train_mode=False,
        interp_shape=None,
    ):
        model.eval()
        per_batch_eval_results = []

        if self.visualize_interval > 0:
            os.makedirs(self.visualize_dir, exist_ok=True)

        for batch_idx, sequence in enumerate(tqdm(test_dataloader)):
            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = sequence["img"]
            if not is_real_data:
                batch_dict["disparity"] = sequence["disp"][:, 0].abs()
                batch_dict["disparity_mask"] = sequence["valid_disp"][:, :1]

                if "mask" in sequence:
                    batch_dict["fg_mask"] = sequence["mask"][:, :1]
                else:
                    batch_dict["fg_mask"] = torch.ones_like(
                        batch_dict["disparity_mask"]
                    )
            elif interp_shape is not None:
                left_video = batch_dict["stereo_video"][:, 0]
                left_video = F.interpolate(
                    left_video, tuple(interp_shape), mode="bilinear"
                )
                right_video = batch_dict["stereo_video"][:, 1]
                right_video = F.interpolate(
                    right_video, tuple(interp_shape), mode="bilinear"
                )
                batch_dict["stereo_video"] = torch.stack([left_video, right_video], 1)

            if train_mode:
                predictions = model.forward_batch_test(batch_dict)
            else:
                predictions = model(batch_dict)

            assert "disparity" in predictions
            # predictions["disparity"] = predictions["disparity"][:, :1].clone().cpu() # Removed to handle video generation

            print(f"Predictions keys: {predictions.keys()}")
            print(f"Disparity shape: {predictions['disparity'].shape}")

            # Always take the first frame of the predicted sequence.
            # Since our custom dataset now slides by 1 and pads the end,
            # Batch i corresponds to Frame i.
            predictions["disparity"] = predictions["disparity"][:, :1].clone().cpu()

            # Crop if padding was applied
            if "original_size" in batch_dict:
                h_orig, w_orig = batch_dict["original_size"][0]
                h_orig = int(h_orig.item())
                w_orig = int(w_orig.item())
                predictions["disparity"] = predictions["disparity"][..., :h_orig, :w_orig]

            if not is_real_data:

                per_batch_eval_results.append((batch_eval_result, seq_length))
                pretty_print_perception_metrics(batch_eval_result)

            if (self.visualize_interval > 0) and (
                batch_idx % self.visualize_interval == 0
            ):
                perception_prediction = PerceptionPrediction()

                pred_disp = predictions["disparity"]
                pred_disp[pred_disp < self.eps] = self.eps

                scale = depth2disparity_scale(
                    sequence["viewpoint"][0][0],
                    sequence["viewpoint"][0][1],
                    torch.tensor([pred_disp.shape[2], pred_disp.shape[3]])[None],
                )

                perception_prediction.depth_map = (scale / pred_disp).cuda()
                perspective_cameras = []
                for cam in sequence["viewpoint"]:
                    perspective_cameras.append(cam[0])

                perception_prediction.perspective_cameras = perspective_cameras

                if "stereo_original_video" in batch_dict:
                    batch_dict["stereo_video"] = batch_dict[
                        "stereo_original_video"
                    ].clone()

                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.cuda()

                visualize_batch(
                    batch_dict,
                    perception_prediction,
                    output_dir=self.visualize_dir,
                    sequence_name=sequence["metadata"][0][0][0],
                    step=step,
                    writer=writer,
                )

            # Save predicted disparities
            if predictions["disparity"].shape[0] > 0:
                disparity_dir = os.path.join(self.exp_dir, "disparities")
                os.makedirs(disparity_dir, exist_ok=True)
                for seq_idx in range(predictions["disparity"].shape[0]):
                    disp = predictions["disparity"][seq_idx, 0].cpu().numpy()
                    # Use batch_idx to name the file, as batch_idx == frame_idx in our new custom scheme
                    if is_real_data:
                         np.save(os.path.join(disparity_dir, f"frame_{batch_idx:04d}.npy"), disp)
                    else:
                         np.save(os.path.join(disparity_dir, f"batch_{batch_idx}_frame_{seq_idx}.npy"), disp)

        return per_batch_eval_results
