# -----------------------------------------------------------------------------
# Copyright 2024 Sony Semiconductor Solutions, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
from typing import Union, Sequence
import warnings

import torch
from torch import nn

from edgemdt_cl.common.box_util import corners_to_centroids, centroids_to_corners
from edgemdt_cl.pytorch.custom_lib import register_op

BOX_DECODE_TORCH_OP = 'faster_rcnn_box_decode'

__all__ = ['FasterRCNNBoxDecode']


class FasterRCNNBoxDecode(nn.Module):
    """
    Box decoding as per Faster R-CNN <https://arxiv.org/abs/1506.01497>.

    Args:
        anchors: Anchors with a shape of (n_boxes, 4) in corner coordinates (y_min, x_min, y_max, x_max).
        scale_factors: Scaling factors in the format (y, x, height, width).
        clip_window: Clipping window in the format (y_min, x_min, y_max, x_max).

    Inputs:
        **rel_codes** (Tensor): Relative codes (encoded offsets) with a shape of (batch, n_boxes, 4) in centroid
                                coordinates (y_center, x_center, h, w).

    Returns:
        Decoded boxes with a shape of (batch, n_boxes, 4) in corner coordinates (y_min, x_min, y_max, x_max).

    Raises:
        ValueError: If provided with invalid arguments or an input tensor with unexpected shape

    Example:
        ```
        from edgemdt_cl.pytorch import FasterRCNNBoxDecode

        box_decode = FasterRCNNBoxDecode(anchors,
                                         scale_factors=(10, 10, 5, 5),
                                         clip_window=(0, 0, 1, 1))
        decoded_boxes = box_decode(rel_codes)
        ```
    """

    def __init__(self, anchors: torch.Tensor, scale_factors: Sequence[Union[float, int]],
                 clip_window: Sequence[Union[float, int]]):
        super().__init__()
        warnings.warn("FasterRCNNBoxDecode is deprecated and will be removed in a future version.", DeprecationWarning)

        if not (len(anchors.shape) == 2 and anchors.shape[-1] == 4):
            raise ValueError(f'Invalid anchors shape {anchors.shape}. Expected shape (n_boxes, 4).')
        self.register_buffer('anchors', anchors)

        if len(scale_factors) != 4:
            raise ValueError(f'Invalid scale factors {scale_factors}. Expected 4 values for (y, x, height, width).')
        self.register_buffer('scale_factors', torch.tensor(scale_factors, dtype=torch.float32, device=anchors.device))

        if len(clip_window) != 4:
            raise ValueError(f'Invalid clip window {clip_window}. Expected 4 values for (y_min, x_min, y_max, x_max).')
        self.register_buffer('clip_window', torch.tensor(clip_window, dtype=torch.float32, device=anchors.device))

    def forward(self, rel_codes: torch.Tensor) -> torch.Tensor:
        return torch.ops.edgemdt.faster_rcnn_box_decode(rel_codes, self.anchors, self.scale_factors, self.clip_window)


######################
# Register custom op #
######################


def _faster_rcnn_box_decode_impl(rel_codes: torch.Tensor, anchors: torch.Tensor, scale_factors: torch.Tensor,
                                 clip_window: torch.Tensor) -> torch.Tensor:
    """ This implementation is intended only to be registered as custom torch and onnxruntime op. """
    if len(rel_codes.shape) != 3 or rel_codes.shape[-1] != 4:
        raise ValueError(f'Invalid input tensor shape {rel_codes.shape}. Expected shape (batch, n_boxes, 4).')

    if rel_codes.shape[-2] != anchors.shape[-2]:
        raise ValueError(f'Mismatch in the number of boxes between input tensor ({rel_codes.shape[-2]}) '
                         f'and anchors ({anchors.shape[-2]})')

    scaled_codes = rel_codes / scale_factors

    a_y_min, a_x_min, a_y_max, a_x_max = torch.unbind(anchors, dim=-1)
    a_y_center, a_x_center, a_h, a_w = corners_to_centroids(a_y_min, a_x_min, a_y_max, a_x_max)

    box_y_center = scaled_codes[..., 0] * a_h + a_y_center
    box_x_center = scaled_codes[..., 1] * a_w + a_x_center
    box_h = torch.exp(scaled_codes[..., 2]) * a_h
    box_w = torch.exp(scaled_codes[..., 3]) * a_w
    box_y_min, box_x_min, box_y_max, box_x_max = centroids_to_corners(box_y_center, box_x_center, box_h, box_w)
    boxes = torch.stack([box_y_min, box_x_min, box_y_max, box_x_max], dim=-1)

    y_low, x_low, y_high, x_high = clip_window
    boxes = torch.clip(boxes, torch.tensor([y_low, x_low, y_low, x_low], device=rel_codes.device),
                       torch.tensor([y_high, x_high, y_high, x_high], device=rel_codes.device))
    return boxes


schema = (BOX_DECODE_TORCH_OP +
          "(Tensor rel_codes, Tensor anchors, Tensor scale_factors, Tensor clip_window) -> Tensor")

register_op(BOX_DECODE_TORCH_OP, schema, _faster_rcnn_box_decode_impl)
