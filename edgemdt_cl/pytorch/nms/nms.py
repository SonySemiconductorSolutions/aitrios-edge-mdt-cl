# -----------------------------------------------------------------------------
# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import NamedTuple, Callable

import torch
from torch import Tensor
import torchvision    # noqa: F401 # needed for torch.ops.torchvision

from edgemdt_cl.pytorch.custom_lib import register_op
from .nms_common import _batch_multiclass_nms, SCORES, LABELS
from edgemdt_cl.pytorch.custom_layer import CustomLayer

MULTICLASS_NMS_TORCH_OP = 'multiclass_nms'

__all__ = ['multiclass_nms', 'NMSResults', 'MulticlassNMS']


class NMSResults(NamedTuple):
    """ Container for non-maximum suppression results """
    boxes: Tensor
    scores: Tensor
    labels: Tensor
    n_valid: Tensor

    # Note: convenience methods below are replicated in each Results container, since NamedTuple supports neither adding
    # new fields in derived classes nor multiple inheritance, and we want it to behave like a tuple, so no dataclasses.
    def detach(self) -> 'NMSResults':
        """ Detach all tensors and return a new object """
        return self.apply(lambda t: t.detach())

    def cpu(self) -> 'NMSResults':
        """ Move all tensors to cpu and return a new object """
        return self.apply(lambda t: t.cpu())

    def apply(self, f: Callable[[Tensor], Tensor]) -> 'NMSResults':
        """ Apply any function to all tensors and return a new object """
        return self.__class__(*[f(t) for t in self])


def multiclass_nms(boxes, scores, score_threshold: float, iou_threshold: float, max_detections: int) -> NMSResults:
    """
    Multi-class non-maximum suppression.
    Detections are returned in descending order of their scores.
    The output tensors always contain a fixed number of detections, as defined by 'max_detections'.
    If fewer detections are selected, the output tensors are zero-padded up to 'max_detections'.

    If you also require the input indices of the selected boxes, see `multiclass_nms_with_indices`.

    Args:
        boxes (Tensor): Input boxes with shape [batch, n_boxes, 4], specified in corner coordinates
                        (x_min, y_min, x_max, y_max). Agnostic to the x-y axes order.
        scores (Tensor): Input scores with shape [batch, n_boxes, n_classes].
        score_threshold (float): The score threshold. Candidates with scores below the threshold are discarded.
        iou_threshold (float): The Intersection Over Union (IOU) threshold for boxes overlap.
        max_detections (int): The number of detections to return.

    Returns:
        'NMSResults' named tuple:
        - boxes: The selected boxes with shape [batch, max_detections, 4].
        - scores: The corresponding scores in descending order with shape [batch, max_detections].
        - labels: The labels for each box with shape [batch, max_detections].
        - n_valid: The number of valid detections out of 'max_detections' with shape [batch, 1]

    Raises:
        ValueError: If provided with invalid arguments or input tensors with unexpected or non-matching shapes.

    Example:
        ```
        from edgemdt_cl.pytorch import multiclass_nms

        # batch size=1, 1000 boxes, 50 classes
        boxes = torch.rand(1, 1000, 4)
        scores = torch.rand(1, 1000, 50)
        res = multiclass_nms(boxes,
                             scores,
                             score_threshold=0.1,
                             iou_threshold=0.6,
                             max_detections=300)
        # res.boxes, res.scores, res.labels, res.n_valid
        ```
    """
    return NMSResults(*torch.ops.edgemdt.multiclass_nms(boxes, scores, score_threshold, iou_threshold, max_detections))


class MulticlassNMS(CustomLayer):
    """
    A torch.nn.Module for multiclass NMS. See multiclass_nms for additional information.

    Usage example:
        batch size=1, 1000 boxes, 50 classes
        boxes = torch.rand(1, 1000, 4)
        scores = torch.rand(1, 1000, 50)
        nms = MulticlassNMS(score_threshold=0.1,
                            iou_threshold=0.6
                            max_detections=300)
        res = nms(boxes, scores)
    """

    def __init__(self, score_threshold: float, iou_threshold: float, max_detections: int):
        """
        Args:
            score_threshold (float): The score threshold. Candidates with scores below the threshold are discarded.
            iou_threshold (float): The Intersection Over Union (IOU) threshold for boxes overlap.
            max_detections (int): The number of detections to return.
        """
        super(MulticlassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor):
        """
        Args:
            boxes (Tensor): Input boxes with shape [batch, n_boxes, 4], specified in corner coordinates
                            (x_min, y_min, x_max, y_max). Agnostic to the x-y axes order.
            scores (Tensor): Input scores with shape [batch, n_boxes, n_classes].

        Returns: 'NMSResults' named tuple:
            - boxes: The selected boxes with shape [batch, max_detections, 4].
            - scores: The corresponding scores in descending order with shape [batch, max_detections].
            - labels: The labels for each box with shape [batch, max_detections].
            - n_valid: The number of valid detections out of 'max_detections' with shape [batch, 1]
        """
        nms = multiclass_nms(boxes=boxes,
                             scores=scores,
                             score_threshold=self.score_threshold,
                             iou_threshold=self.iou_threshold,
                             max_detections=self.max_detections)
        return nms


######################
# Register custom op #
######################


def _multiclass_nms_impl(boxes: torch.Tensor, scores: torch.Tensor, score_threshold: float, iou_threshold: float,
                         max_detections: int) -> NMSResults:
    """ This implementation is intended only to be registered as custom torch and onnxruntime op.
        NamedTuple is used for clarity, it is not preserved when run through torch / onnxruntime op. """
    res, valid_dets = _batch_multiclass_nms(boxes,
                                            scores,
                                            score_threshold=score_threshold,
                                            iou_threshold=iou_threshold,
                                            max_detections=max_detections)
    return NMSResults(boxes=res[..., :4],
                      scores=res[..., SCORES],
                      labels=res[..., LABELS].to(torch.int64),
                      n_valid=valid_dets.to(torch.int64))


schema = (MULTICLASS_NMS_TORCH_OP +
          "(Tensor boxes, Tensor scores, float score_threshold, float iou_threshold, SymInt max_detections) "
          "-> (Tensor, Tensor, Tensor, Tensor)")

register_op(MULTICLASS_NMS_TORCH_OP, schema, _multiclass_nms_impl)
