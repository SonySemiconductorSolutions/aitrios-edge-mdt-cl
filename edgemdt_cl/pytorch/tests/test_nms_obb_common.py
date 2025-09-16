# -----------------------------------------------------------------------------
#  Copyright 2025 Sony Semiconductor Solutions. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -----------------------------------------------------------------------------
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from edgemdt_cl.pytorch.nms_obb import nms_obb_common
from edgemdt_cl.pytorch.nms_obb.nms_obb_common import LABELS, INDICES, SCORES, ANGLES


def generate_random_inputs_obb(batch: Optional[int], n_boxes, n_classes, seed=None):
    boxes_shape = (batch, n_boxes, 4) if batch else (n_boxes, 4)
    scores_shape = (batch, n_boxes, n_classes) if batch else (n_boxes, n_classes)
    angles_shape = (batch, n_boxes, 1) if batch else (n_boxes, 1)
    if seed:
        torch.random.manual_seed(seed)
    boxes = torch.rand(*boxes_shape)
    scores = torch.rand(*scores_shape)
    angles = torch.rand(*angles_shape)
    return boxes, scores, angles


class TestNMSOBBCommon:

    @pytest.mark.parametrize('max_detections', [3, 6, 10])
    def test_image_multiclass_nms_obb(self, max_detections):
        
        boxes = Tensor([[10, 10, 4, 3],
                        [10.5, 10, 4.5, 2.5],
                        [20, 20, 1, 5],
                        [20.5, 20, 1.5, 5],
                        [30, 30, 1, 5],
                        [30, 31, 1, 6],
                        [40, 40, 2, 2]])    # yapf: disable
        scores = Tensor([[0.2, 0.1, 0.25],
                         [0.2, 0.1, 0.3],
                         [0.3, 0.2, 0.05],
                         [0.1, 0.4, 0.05],
                         [0.05, 0.1, 0.5],
                         [0.1, 0.15, 0.55],
                         [0.15, 0.1, 0.05]])    # yapf: disable
        angles = Tensor([[0.78],
                         [0.785],
                         [1.5],
                         [1.3],
                         [0.1],
                         [2.3],
                         [-0.5]])    # yapf: disable
        score_threshold = 0.1
        iou_threshold = 0.6

        ret, ret_valid_dets = nms_obb_common._image_multiclass_nms_obb(boxes, 
                                                                       scores, 
                                                                       angles, 
                                                                       score_threshold=score_threshold,
                                                                       iou_threshold=iou_threshold,
                                                                       max_detections=max_detections)              
        assert ret.shape == (max_detections, 8)
        exp_valid_dets = min(6, max_detections)

        exp_boxes = Tensor([[30.0, 31.0,  1.0,  6.0],
                            [30.0, 30.0,  1.0,  5.0],
                            [20.5, 20.0,  1.5,  5.0],
                            [10.5, 10.0,  4.5,  2.5],
                            [20.0, 20.0,  1.0,  5.0],
                            [40.0, 40.0,  2.0,  2.0]])  # yapf: disable
        exp_scores = Tensor([0.55, 0.5, 0.4, 0.3, 0.3, 0.15])
        exp_labels = Tensor([2, 2, 1, 2, 0, 0])
        exp_angles = Tensor([2.3, 0.1, 1.3, 0.785, 1.5, -0.5])
        exp_indices = Tensor([5, 4, 3, 1, 2, 6])

        # check for boxes
        assert torch.equal(ret[:, :4][:exp_valid_dets], exp_boxes[:exp_valid_dets])
        assert torch.all(ret[:, :4][exp_valid_dets:] == 0)
        # check for scores
        assert torch.equal(ret[:, SCORES][:exp_valid_dets], exp_scores[:exp_valid_dets])
        assert torch.all(ret[:, SCORES][exp_valid_dets:] == 0)
        # check for labels
        assert torch.equal(ret[:, LABELS][:exp_valid_dets], exp_labels[:exp_valid_dets])
        assert torch.all(ret[:, LABELS][exp_valid_dets:] == 0)
        # check for angles
        assert torch.equal(ret[:, ANGLES][:exp_valid_dets], exp_angles[:exp_valid_dets])
        assert torch.all(ret[:, ANGLES][exp_valid_dets:] == 0)
        # check for indices
        assert torch.equal(ret[:, INDICES][:exp_valid_dets], exp_indices[:exp_valid_dets])
        assert torch.all(ret[:, INDICES][exp_valid_dets:] == 0)

        assert ret_valid_dets == exp_valid_dets

    def test_image_multiclass_nms_obb_no_valid_boxes(self):
        boxes, scores, angles = generate_random_inputs_obb(None, 100, 20)
        scores = 0.01 * scores
        score_threshold = 0.5
        res, n_valid_dets = nms_obb_common._image_multiclass_nms_obb(boxes,
                                                                     scores,
                                                                     angles,
                                                                     score_threshold=score_threshold,
                                                                     iou_threshold=0.1,
                                                                     max_detections=200)
        assert torch.equal(res, torch.zeros(200, 8))
        assert n_valid_dets == 0

    def test_image_multiclass_nms_obb_single_class(self):
        boxes, scores, angles = generate_random_inputs_obb(None, 100, 1)
        res, n_valid_dets = nms_obb_common._image_multiclass_nms_obb(boxes,
                                                                     scores,
                                                                     angles,
                                                                     score_threshold=0.1,
                                                                     iou_threshold=0.1,
                                                                     max_detections=50)
        assert res.shape == (50, 8)
        assert n_valid_dets > 0
        assert torch.equal(res[:n_valid_dets, LABELS], torch.zeros((n_valid_dets, )))

    def test_batch_multiclass_nms_obb(self, mocker):
        input_boxes, input_scores, input_angles = generate_random_inputs_obb(batch=3, n_boxes=20, n_classes=10)
        max_dets = 5

        # these numbers don't really make sense as nms outputs, but we don't really care, we only want to test
        # that outputs are combined correctly
        img_nms_ret = torch.rand(3, max_dets, 8)
        img_nms_ret[..., LABELS] = torch.randint(0, 10, (3, max_dets), dtype=torch.float32)
        img_nms_ret[..., INDICES] = torch.randint(0, 20, (3, max_dets), dtype=torch.float32)

        ret_valid_dets = Tensor([[5], [4], [3]])
        # each time the function is called, next value in the list returned
        images_ret = [(img_nms_ret[i], ret_valid_dets[i]) for i in range(3)]
        mock = mocker.patch('edgemdt_cl.pytorch.nms_obb.nms_obb_common._image_multiclass_nms_obb',
                            Mock(side_effect=lambda *args, **kwargs: images_ret.pop(0)))

        res, n_valid = nms_obb_common._batch_multiclass_nms_obb(input_boxes,
                                                                input_scores,
                                                                input_angles,
                                                                score_threshold=0.1,
                                                                iou_threshold=0.6,
                                                                max_detections=5)

        # check each invocation
        for i, call_args in enumerate(mock.call_args_list):
            assert torch.equal(call_args.args[0], input_boxes[i]), i
            assert torch.equal(call_args.args[1], input_scores[i]), i
            assert torch.equal(call_args.args[2], input_angles[i]), i
            assert call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5), i

        assert torch.equal(res, img_nms_ret)
        assert torch.equal(n_valid, ret_valid_dets)
