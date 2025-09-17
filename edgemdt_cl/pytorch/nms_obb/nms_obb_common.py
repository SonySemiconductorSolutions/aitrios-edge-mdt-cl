# -----------------------------------------------------------------------------
# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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
ANGLES = 6
INDICES = 7


def _batch_multiclass_nms_obb(boxes: Union[Tensor, np.ndarray], scores: Union[Tensor, np.ndarray],
                              angles: Union[Tensor, np.ndarray], score_threshold: float, iou_threshold: float,
                              max_detections: int) -> Tuple[Tensor, Tensor]:
    """
    Performs multi-class non-maximum suppression for oriented bounding box on a batch of images

    Args:
        boxes: input boxes of shape [batch, n_boxes, 4]
        scores: input scores of shape [batch, n_boxes, n_classes]
        angles: input angles of shape [batch, n_boxes, 1]
        score_threshold: score threshold
        iou_threshold: intersection over union threshold
        max_detections: fixed number of detections to return

    Returns:
        A tuple of two tensors:
        - results: A tensor of shape [batch, max_detections, 8]
                   containing the results of multiclass nms for oriented bounding box.
        - valid_dets: A tensor of shape [batch, 1] containing the number of valid detections.

    """
    # this is needed for onnxruntime implementation
    if not isinstance(boxes, Tensor):
        boxes = Tensor(boxes)
    if not isinstance(scores, Tensor):
        scores = Tensor(scores)
    if not isinstance(angles, Tensor):
        angles = Tensor(angles)

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
    if angles.ndim != 3 or angles.shape[-1] != 1:
        raise ValueError(f'Invalid input angles shape {angles.shape}. Expected shape (batch, n_boxes, 1).')
    if (boxes.shape[-2] != scores.shape[-2]) or (boxes.shape[-2] != angles.shape[-2]):
        raise ValueError(f'Mismatch in the number of boxes between input boxes ({boxes.shape[-2]}) '
                         f'and scores ({scores.shape[-2]}), angles ({angles.shape[-2]}) ')

    batch = boxes.shape[0]
    results = torch.zeros((batch, max_detections, 8), device=boxes.device)
    valid_dets = torch.zeros((batch, 1), device=boxes.device)
    for i in range(batch):
        results[i], valid_dets[i] = _image_multiclass_nms_obb(boxes[i],
                                                              scores[i],
                                                              angles[i],
                                                              score_threshold=score_threshold,
                                                              iou_threshold=iou_threshold,
                                                              max_detections=max_detections)
    return results, valid_dets


def _image_multiclass_nms_obb(boxes: Tensor, scores: Tensor, angles: Tensor, score_threshold: float,
                              iou_threshold: float, max_detections: int) -> Tuple[Tensor, int]:
    """
    Performs multi-class non-maximum suppression for oriented bounding box on a single image.
    This algorithm is referenced in https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        boxes: input boxes of shape [n_boxes, 4]
        scores: input scores of shape [n_boxes, n_classes]
        angles: input angles of shape [n_boxes, 1]
        score_threshold: score threshold
        iou_threshold: intersection over union threshold
        max_detections: fixed number of detections to return

    Returns:
        A tensor 'out' of shape [max_detections, 8] and the number of valid detections.
        out[:, :4] contains the selected boxes.
        out[:, 4] contains the scores for the selected boxes.
        out[:, 5] contains the labels for the selected boxes.
        out[:, 6] contains the angles for the selected boxes.
        out[:, 7] contains indices of input boxes that have been selected.

    """
    score_th_idx = scores.amax(1) > score_threshold

    boxes = boxes[score_th_idx]
    scores = scores[score_th_idx]
    angles = angles[score_th_idx]

    out = torch.zeros(max_detections, 8, device=boxes.device)
    if boxes.size(0) == 0:
        return out, 0

    class_scores, class_labels = scores.max(dim=1, keepdim=True)

    offsets = class_labels.float() * (boxes.max() + 1)
    boxes[:, :2] = boxes[:, :2] + offsets

    score_sort_idx = torch.argsort(class_scores.squeeze_(-1), descending=True)
    sorted_boxes = boxes[score_sort_idx]
    sorted_angles = angles[score_sort_idx]

    ious_th_idx = _calc_iou(sorted_boxes, sorted_angles, iou_threshold)
    idxs = score_sort_idx[ious_th_idx]

    if idxs.size(0) > max_detections:
        idxs = idxs[:max_detections]

    class_labels = class_labels.squeeze_(-1)
    angles = angles.squeeze_(-1)

    boxes[:, :2] = boxes[:, :2] - offsets

    out[:idxs.size(0), :4] = boxes[idxs]
    out[:idxs.size(0), SCORES] = class_scores[idxs]
    out[:idxs.size(0), LABELS] = class_labels[idxs]
    out[:idxs.size(0), ANGLES] = angles[idxs]
    out[:idxs.size(0), INDICES] = idxs
    valid_dets = idxs.size(0)

    return out, valid_dets


def _calc_iou(boxes: Tensor, angles: Tensor, iou_threshold: float) -> Tensor:
    """
    Calculate intersection over union (IoU) from bounding boxes and angles.
    Generate the indices to remove overlapping boxes using IoU threshold.

    Args:
        boxes: input boxes of shape [n_boxes, 4]
        angles: input angles of shape [n_boxes, 1]
        iou_threshold: intersection over union threshold

    Returns:
        A tensor 'ious_th_idx' represents the indices of boxes to keep.
    """
    eps = 1e-10
    cx1, cy1 = boxes[:, 0:1], boxes[:, 1:2]
    w1, h1 = boxes[:, 2:3].pow(2) / 12, boxes[:, 3:4].pow(2) / 12
    cos1, sin1 = angles.cos(), angles.sin()

    a1 = w1 * cos1.pow(2) + h1 * sin1.pow(2)
    b1 = w1 * sin1.pow(2) + h1 * cos1.pow(2)
    c1 = (w1 - h1) * cos1 * sin1

    cx2, cy2 = cx1.view(1, -1), cy1.view(1, -1)
    a2 = a1.view(1, -1)
    b2 = b1.view(1, -1)
    c2 = c1.view(1, -1)

    top_B1 = (a1 + a2) * (cy1 - cy2).pow(2) + (b1 + b2) * (cx1 - cx2).pow(2) + 2.0 * (c1 + c2) * (cx2 - cx1) * (cy1 -
                                                                                                                cy2)
    bot_B1 = 4.0 * ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    B1 = top_B1 / bot_B1

    top_B2 = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)
    bot_B2 = 4.0 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps
    B2 = (top_B2 / bot_B2).log() / 2.0

    bd = (B1 + B2).clamp(eps, 1000.0)
    bc = (-bd).exp()
    hd = (1.0 - bc).clamp_(0).sqrt()
    ious = 1.0 - hd
    ious = ious.triu_(diagonal=1)
    ious_th_idx = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).view(-1)

    return ious_th_idx
