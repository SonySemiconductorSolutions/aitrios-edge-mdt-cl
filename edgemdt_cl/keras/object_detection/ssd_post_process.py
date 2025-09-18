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

from typing import Sequence, Tuple, Union, List
import dataclasses

import tensorflow as tf
import numpy as np

from edgemdt_cl.keras.base_custom_layer import CustomLayer
from edgemdt_cl.keras.object_detection import FasterRCNNBoxDecode, ScoreConverter
from edgemdt_cl.keras.custom_objects import register_layer


@dataclasses.dataclass
class SSDPostProcessCfg:
    anchors: Union[np.ndarray, tf.Tensor, List[List[float]]]
    scale_factors: Sequence[Union[float, int]]
    clip_size: Sequence[Union[float, int]]
    score_converter: Union[ScoreConverter, str]
    score_threshold: float
    iou_threshold: float
    max_detections: float
    remove_background: bool

    def as_dict(self):
        return dataclasses.asdict(self)


@register_layer
class SSDPostProcess(CustomLayer):
    """
    SSD Post Processing, based on <https://arxiv.org/abs/1512.02325>.

    Args:
        anchors (Tensor | np.ndarray): Anchors with a shape of (n_boxes, 4) in corner coordinates
                                       (y_min, x_min, y_max, x_max).
        scale_factors (list | tuple): Box decoding scaling factors in the format (y, x, height, width).
        clip_size (list | tuple): Clipping size in the format (height, width). The decoded boxes are clipped to the
                                  range y=[0, height] and x=[0, width]. Typically, the clipping size is (1, 1) for
                                  normalized boxes and the image size for boxes in pixel coordinates.
        score_converter (ScoreConverter): Conversion to apply to the input logits (sigmoid, softmax, or linear).
        score_threshold (float): Score threshold for non-maximum suppression.
        iou_threshold (float): Intersection over union threshold for non-maximum suppression.
        max_detections (int): The number of detections to return.
        remove_background (bool) : If True, the first class is removed from the input scores (after the score_converter
                                   is applied).

    Inputs:
        A list or tuple of:
        - **rel_codes** (Tensor): Relative codes (encoded offsets) with a shape of (batch, n_boxes, 4) in centroid
                            coordinates (y_center, x_center, w, h).
        - **scores** (Tensor): Scores or logits with a shape of (batch, n_boxes, n_labels).

    Returns:
        'CombinedNonMaxSuppression' named tuple:
        - nmsed_boxes: Selected boxes sorted by scores in descending order, with a shape of
                         (batch, max_detections, 4),in corner coordinates (y_min, x_min, y_max, x_max).
        - nmsed_scores: Scores corresponding to the selected boxes, with a shape of (batch, max_detections).
        - nmsed_classes: Labels corresponding to the selected boxes, with a shape of (batch, max_detections).
                           Each label corresponds to the class index of the selected score in the input scores.
        - valid_detections: The number of valid detections out of max_detections.

    Raises:
        ValueError: If provided with invalid arguments or input tensors with unexpected or non-matching shapes.

    Example:
        ```
        from edgemdt_cl.keras import SSDPostProcessing, ScoreConverter

        post_process = SSDPostProcess(anchors=anchors,
                                      scale_factors=(10, 10, 5, 5),
                                      clip_size=(320, 320),
                                      score_converter=ScoreConverter.SIGMOID,
                                      score_threshold=0.01,
                                      iou_threshold=0.6,
                                      max_detections=200,
                                      remove_background=True)
        res = post_process([rel_codes, logits])
        boxes = res.nmsed_boxes
        ```
    """

    def __init__(self,
                 anchors: Union[np.ndarray, tf.Tensor, List[List[float]]],
                 scale_factors: Sequence[Union[int, float]],
                 clip_size: Sequence[Union[int, float]],
                 score_converter: Union[ScoreConverter, str],
                 score_threshold: float,
                 iou_threshold: float,
                 max_detections: int,
                 remove_background: bool = False,
                 **kwargs):
        """ """
        super().__init__(**kwargs)

        if not 0 <= score_threshold <= 1:
            raise ValueError(f'Invalid score_threshold {score_threshold} not in range [0, 1]')
        if not 0 <= iou_threshold <= 1:
            raise ValueError(f'Invalid iou_threshold {iou_threshold} not in range [0, 1]')
        if max_detections <= 0:
            raise ValueError(f'Invalid non-positive max_detections {max_detections}')

        self.cfg = SSDPostProcessCfg(anchors=anchors,
                                     scale_factors=scale_factors,
                                     clip_size=clip_size,
                                     score_converter=score_converter,
                                     score_threshold=score_threshold,
                                     iou_threshold=iou_threshold,
                                     max_detections=max_detections,
                                     remove_background=remove_background)
        self._box_decode = FasterRCNNBoxDecode(anchors, scale_factors, (0, 0, *clip_size))

    def call(self, inputs: Sequence[tf.Tensor], *args, **kwargs) -> Tuple[tf.Tensor]:
        """ """
        rel_codes, scores = inputs
        if len(rel_codes.shape) != 3 and rel_codes.shape[-1] != 4:
            raise ValueError(f'Invalid input offsets shape {rel_codes.shape}. '
                             f'Expected shape (batch, n_boxes, 4).')
        if len(scores.shape) != 3:
            raise ValueError(f'Invalid input scores shape {scores.shape}. '
                             f'Expected shape (batch, n_boxes, n_labels).')
        if rel_codes.shape[-2] != scores.shape[-2]:
            raise ValueError(f'Mismatch in the number of boxes between input codes ({rel_codes.shape[-2]}) '
                             f'and input scores ({scores.shape[-2]}).')

        if self.cfg.score_converter != ScoreConverter.LINEAR:
            scores = tf.keras.layers.Activation(self.cfg.score_converter)(scores)

        if self.cfg.remove_background:
            scores = tf.slice(scores, begin=[0, 0, 1], size=[-1, -1, -1])

        decoded_boxes = self._box_decode(rel_codes)
        # when decoded_boxes.shape[-2]==1, nms uses same boxes for all classes
        decoded_boxes = tf.expand_dims(decoded_boxes, axis=-2)

        outputs = tf.image.combined_non_max_suppression(decoded_boxes,
                                                        scores,
                                                        max_output_size_per_class=self.cfg.max_detections,
                                                        max_total_size=self.cfg.max_detections,
                                                        iou_threshold=self.cfg.iou_threshold,
                                                        score_threshold=self.cfg.score_threshold,
                                                        pad_per_class=False,
                                                        clip_boxes=False)
        return outputs

    def get_config(self) -> dict:
        """ """
        config = super().get_config()
        d = self.cfg.as_dict()
        d['anchors'] = tf.constant(d['anchors']).numpy().tolist()
        config.update(d)
        return config
