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
from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor

SCORES = 4
LABELS = 5
INDICES = 6


def _batch_multiclass_nms(boxes: Union[Tensor, np.ndarray], scores: Union[Tensor, np.ndarray], score_threshold: float,
                          iou_threshold: float, max_detections: int) -> Tuple[Tensor, Tensor]:
    """
    Performs multi-class non-maximum suppression on a batch of images

    Args:
        boxes: input boxes of shape [batch, n_boxes, 4]
        scores: input scores of shape [batch, n_boxes, n_classes]
        score_threshold: score threshold
        iou_threshold: intersection over union threshold
        max_detections: fixed number of detections to return

    Returns:
        A tuple of two tensors:
        - results: A tensor of shape [batch, max_detections, 7] containing the results of multiclass nms.
        - valid_dets: A tensor of shape [batch, 1] containing the number of valid detections.

    """
    # this is needed for onnxruntime implementation
    if not isinstance(boxes, Tensor):
        boxes = Tensor(boxes)
    if not isinstance(scores, Tensor):
        scores = Tensor(scores)

    if not 0 <= score_threshold <= 1:
        raise ValueError(f'Invalid score_threshold {score_threshold} not in range [0, 1]')
    if not 0 <= iou_threshold <= 1:
        raise ValueError(f'Invalid iou_threshold {iou_threshold} not in range [0, 1]')
    if max_detections <= 0:
        raise ValueError(f'Invalid non-positive max_detections {max_detections}')

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f'Invalid input boxes shape {boxes.shape}. Expected shape (batch, n_boxes, 4).')
    if scores.ndim != 3:
        raise ValueError(f'Invalid input scores shape {scores.shape}. Expected shape (batch, n_boxes, n_classes).')
    if boxes.shape[-2] != scores.shape[-2]:
        raise ValueError(f'Mismatch in the number of boxes between input boxes ({boxes.shape[-2]}) '
                         f'and scores ({scores.shape[-2]})')

    batch = boxes.shape[0]
    results = torch.zeros((batch, max_detections, 7), device=boxes.device)
    valid_dets = torch.zeros((batch, 1), device=boxes.device)
    for i in range(batch):
        results[i], valid_dets[i] = _image_multiclass_nms(boxes[i],
                                                          scores[i],
                                                          score_threshold=score_threshold,
                                                          iou_threshold=iou_threshold,
                                                          max_detections=max_detections)

    return results, valid_dets


def _image_multiclass_nms(boxes: Tensor, scores: Tensor, score_threshold: float, iou_threshold: float,
                          max_detections: int) -> Tuple[Tensor, int]:
    """
    Performs multi-class non-maximum suppression on a single image

    Args:
        boxes: input boxes of shape [n_boxes, 4]
        scores: input scores of shape [n_boxes, n_classes]
        score_threshold: score threshold
        iou_threshold: intersection over union threshold
        max_detections: fixed number of detections to return

    Returns:
        A tensor 'out' of shape [max_detections, 7] and the number of valid detections.
        out[:, :4] contains the selected boxes.
        out[:, 4] contains the scores for the selected boxes.
        out[:, 5] contains the labels for the selected boxes.
        out[:, 6] contains indices of input boxes that have been selected.

    """
    x = _convert_inputs(boxes, scores, score_threshold)
    out = torch.zeros(max_detections, 7, device=boxes.device)
    if x.size(0) == 0:
        return out, 0
    idxs = _nms_with_class_offsets(x[:, :6], iou_threshold=iou_threshold)
    idxs = idxs[:max_detections]
    valid_dets = idxs.numel()
    out[:valid_dets] = x[idxs]
    return out, valid_dets


def _convert_inputs(boxes: Tensor, scores: Tensor, score_threshold: float) -> Tensor:
    """
    Converts inputs into a tensor of candidates and filters out boxes with score below the threshold.

    Args:
        boxes: input boxes of shape [n_boxes, 4]
        scores: input scores of shape [n_boxes, n_classes]
        score_threshold: score threshold for nms candidates

    Returns:
        A tensor of shape [m, 7] containing m nms candidates above the score threshold.
        x[:, :4] contains the boxes with replication for different labels
        x[:, 4] contains the scores
        x[:, 5] contains the labels indices (label i corresponds to input scores[:, i])
        x[:, 6] contains the input boxes indices (candidate x[i, :] corresponds to input box boxes[x[i, 6]]).
            """
    n_boxes, n_classes = scores.shape
    scores_mask = scores > score_threshold
    box_indices = torch.arange(n_boxes, device=boxes.device).unsqueeze(1).expand(-1, n_classes)[scores_mask]
    x = torch.empty((box_indices.numel(), 7), device=boxes.device)
    x[:, :4] = boxes[box_indices]
    x[:, SCORES] = scores[scores_mask]
    x[:, LABELS] = torch.arange(n_classes, device=boxes.device).unsqueeze(0).expand(n_boxes, -1)[scores_mask]
    x[:, INDICES] = box_indices
    return x


def _nms_with_class_offsets(x: Tensor, iou_threshold: float) -> Tensor:
    """
    Multiclass NMS implementation using the single class torchvision op.
    Boxes of each class are shifted so that there is no intersection between boxes of different classes
    (similarly to torchvision batched_nms trick).

    Args:
        x: nms candidates of shape [n, 6] ([:,:4] boxes, [:, 4] scores, [:, 5] labels)
        iou_threshold: intersection over union threshold

    Returns:
        Indices of the selected candidates
    """
    assert x.shape[1] == 6
    offsets = x[:, LABELS:] * (x[:, :4].max() + 1)
    shifted_boxes = x[:, :4] + offsets
    return torch.ops.torchvision.nms(shifted_boxes, x[:, SCORES], iou_threshold)
