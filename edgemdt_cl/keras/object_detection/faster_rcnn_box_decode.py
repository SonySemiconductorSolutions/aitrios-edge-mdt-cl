# -----------------------------------------------------------------------------
# Copyright 2023 Sony Semiconductor Solutions, Inc. All rights reserved.
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

from typing import Sequence, Union, List

import tensorflow as tf
import numpy as np

from edgemdt_cl.common.box_util import corners_to_centroids, centroids_to_corners
from edgemdt_cl.keras.base_custom_layer import CustomLayer
from edgemdt_cl.keras.custom_objects import register_layer


@register_layer
class FasterRCNNBoxDecode(CustomLayer):
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
        from edgemdt_cl.keras import FasterRCNNBoxDecode

        box_decode = FasterRCNNBoxDecode(anchors,
                                         scale_factors=(10, 10, 5, 5),
                                         clip_window=(0, 0, 1, 1))
        decoded_boxes = box_decode(rel_codes)
        ```
    """

    def __init__(self, anchors: Union[np.ndarray, tf.Tensor, List[List[float]]],
                 scale_factors: Sequence[Union[float, int]], clip_window: Sequence[Union[float, int]], **kwargs):
        super().__init__(**kwargs)
        anchors = tf.constant(anchors)
        if not (len(anchors.shape) == 2 and anchors.shape[-1] == 4):
            raise ValueError(f'Invalid anchors shape {anchors.shape}. Expected shape (n_boxes, 4).')
        self.anchors = anchors

        if len(scale_factors) != 4:
            raise ValueError(f'Invalid scale factors {scale_factors}. Expected 4 values for (y, x, height, width).')
        self.scale_factors = tf.constant(scale_factors, dtype=tf.float32)

        if len(clip_window) != 4:
            raise ValueError(f'Invalid clip window {clip_window}. Expected 4 values for (y_min, x_min, y_max, x_max).')
        self.clip_window = clip_window

    def call(self, rel_codes: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """ """
        if len(rel_codes.shape) != 3 or rel_codes.shape[-1] != 4:
            raise ValueError(f'Invalid input tensor shape {rel_codes.shape}. Expected shape (batch, n_boxes, 4).')
        if rel_codes.shape[-2] != self.anchors.shape[-2]:
            raise ValueError(f'Mismatch in the number of boxes between input tensor ({rel_codes.shape[-2]}) '
                             f'and anchors ({self.anchors.shape[-2]})')

        scaled_codes = rel_codes / self.scale_factors

        a_y_min, a_x_min, a_y_max, a_x_max = tf.unstack(self.anchors, axis=-1)
        a_y_center, a_x_center, a_h, a_w = corners_to_centroids(a_y_min, a_x_min, a_y_max, a_x_max)

        box_y_center = scaled_codes[..., 0] * a_h + a_y_center
        box_x_center = scaled_codes[..., 1] * a_w + a_x_center
        box_h = tf.exp(scaled_codes[..., 2]) * a_h
        box_w = tf.exp(scaled_codes[..., 3]) * a_w
        box_y_min, box_x_min, box_y_max, box_x_max = centroids_to_corners(box_y_center, box_x_center, box_h, box_w)
        boxes = tf.stack([box_y_min, box_x_min, box_y_max, box_x_max], axis=-1)

        y_low, x_low, y_high, x_high = self.clip_window
        boxes = tf.clip_by_value(boxes, [y_low, x_low, y_low, x_low], [y_high, x_high, y_high, x_high])
        return boxes

    def get_config(self) -> dict:
        """ """
        config = super().get_config()
        config.update({
            'anchors': self.anchors.numpy().tolist(),
            'scale_factors': self.scale_factors.numpy().tolist(),
            'clip_window': self.clip_window,
        })
        return config
