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
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from edgemdt_cl.pytorch.nms import nms_common


def generate_random_inputs(batch: Optional[int], n_boxes, n_classes, seed=None):
    boxes_shape = (batch, n_boxes, 4) if batch else (n_boxes, 4)
    scores_shape = (batch, n_boxes, n_classes) if batch else (n_boxes, n_classes)
    if seed:
        torch.random.manual_seed(seed)
    boxes = torch.rand(*boxes_shape)
    boxes[..., 0], boxes[..., 2] = torch.aminmax(boxes[..., (0, 2)], dim=-1)
    boxes[..., 1], boxes[..., 3] = torch.aminmax(boxes[..., (1, 3)], dim=-1)
    scores = torch.rand(*scores_shape)
    return boxes, scores


class TestNMSCommon:

    def test_flatten_image_inputs(self):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.11, 0.21, 0.31, 0.41],
                        [0.12, 0.22, 0.32, 0.42]])    # yapf: disable
        scores = Tensor([[0.15, 0.25, 0.35, 0.45],
                         [0.16, 0.26, 0.11, 0.46],
                         [0.1, 0.27, 0.37, 0.47]])    # yapf: disable
        x = nms_common._convert_inputs(boxes, scores, score_threshold=0.11)
        assert x.shape == (10, 7)
        flat_boxes, flat_scores, labels, input_box_indices = x[:, :4], x[:, 4], x[:, 5], x[:, 6]
        assert flat_boxes.shape == (10, 4)
        assert flat_scores.shape == labels.shape == input_box_indices.shape == (10, )
        assert torch.equal(labels, Tensor([0, 1, 2, 3, 0, 1, 3, 1, 2, 3]))
        for i in range(4):
            assert torch.equal(flat_boxes[i], boxes[0]), i
        for i in range(4, 7):
            assert torch.equal(flat_boxes[i], boxes[1]), i
        for i in range(7, 10):
            assert torch.equal(flat_boxes[i], boxes[2]), i
        assert torch.equal(flat_scores, Tensor([0.15, 0.25, 0.35, 0.45, 0.16, 0.26, 0.46, 0.27, 0.37, 0.47]))
        assert torch.equal(input_box_indices, Tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))

    def test_nms_with_class_offsets(self):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8],
                        [0.5, 0.6, 0.7, 0.8],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4]])  # yapf: disable
        scores = Tensor([0.25, 0.15, 0.3, 0.45, 0.5, 0.4])
        labels = Tensor([1, 0, 1, 2, 2, 1])
        x = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        iou_threshold = 0.5
        ret_idxs = nms_common._nms_with_class_offsets(x, iou_threshold)
        assert torch.equal(ret_idxs, Tensor([4, 3, 5, 2, 1]))

    @pytest.mark.parametrize('max_detections', [3, 6, 10])
    # mock is to test our logic, and no mock is for integration sanity
    @pytest.mark.parametrize('mock_tv_op', [True, False])
    def test_image_multiclass_nms(self, mocker, max_detections, mock_tv_op):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8]])    # yapf: disable
        scores = Tensor([[0.2, 0.109, 0.3, 0.12],
                         [0.111, 0.5, 0.05, 0.4]])    # yapf: disable
        score_threshold = 0.11
        iou_threshold = 0.61
        if mock_tv_op:
            nms_mock = mocker.patch('edgemdt_cl.pytorch.nms.nms_common._nms_with_class_offsets',
                                    Mock(return_value=Tensor([4, 5, 1, 0, 2, 3]).to(torch.int64)))
        ret, ret_valid_dets = nms_common._image_multiclass_nms(boxes,
                                                               scores,
                                                               score_threshold=score_threshold,
                                                               iou_threshold=iou_threshold,
                                                               max_detections=max_detections)
        if mock_tv_op:
            assert nms_mock.call_args.args[0].shape == (6, 6)
            assert torch.equal(nms_mock.call_args.args[0][:, :4],
                               Tensor([[0.1, 0.2, 0.3, 0.4],
                                       [0.1, 0.2, 0.3, 0.4],
                                       [0.1, 0.2, 0.3, 0.4],
                                       [0.5, 0.6, 0.7, 0.8],
                                       [0.5, 0.6, 0.7, 0.8],
                                       [0.5, 0.6, 0.7, 0.8]]))    # yapf: disable
            assert torch.equal(nms_mock.call_args.args[0][:, 4], Tensor([0.2, 0.3, 0.12, 0.111, 0.5, 0.4]))
            assert torch.equal(nms_mock.call_args.args[0][:, 5], Tensor([0, 2, 3, 0, 1, 3]))
            assert nms_mock.call_args.kwargs == {'iou_threshold': iou_threshold}

        assert ret.shape == (max_detections, 7)
        exp_valid_dets = min(6, max_detections)
        assert torch.equal(ret[:, :4][:exp_valid_dets],
                           Tensor([[0.5, 0.6, 0.7, 0.8],
                                   [0.5, 0.6, 0.7, 0.8],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])[:exp_valid_dets])  # yapf: disable
        assert torch.all(ret[:, :4][exp_valid_dets:] == 0)
        assert torch.equal(ret[:, 4][:exp_valid_dets], Tensor([0.5, 0.4, 0.3, 0.2, 0.12, 0.111])[:exp_valid_dets])
        assert torch.all(ret[:, 4][exp_valid_dets:] == 0)
        assert torch.equal(ret[:, 5][:exp_valid_dets], Tensor([1, 3, 2, 0, 3, 0])[:exp_valid_dets])
        assert torch.all(ret[:, 5][exp_valid_dets:] == 0)
        assert torch.equal(ret[:, 6][:exp_valid_dets], Tensor([1, 1, 0, 0, 0, 1])[:exp_valid_dets])
        assert torch.all(ret[:, 6][exp_valid_dets:] == 0)
        assert ret_valid_dets == exp_valid_dets

    def test_image_multiclass_nms_no_valid_boxes(self):
        boxes, scores = generate_random_inputs(None, 100, 20)
        scores = 0.5 * scores
        score_threshold = 0.51
        res, n_valid_dets = nms_common._image_multiclass_nms(boxes,
                                                             scores,
                                                             score_threshold=score_threshold,
                                                             iou_threshold=0.1,
                                                             max_detections=200)
        assert torch.equal(res, torch.zeros(200, 7))
        assert n_valid_dets == 0

    def test_image_multiclass_nms_single_class(self):
        boxes, scores = generate_random_inputs(None, 100, 1)
        res, n_valid_dets = nms_common._image_multiclass_nms(boxes,
                                                             scores,
                                                             score_threshold=0.1,
                                                             iou_threshold=0.1,
                                                             max_detections=50)
        assert res.shape == (50, 7)
        assert n_valid_dets > 0
        assert torch.equal(res[:n_valid_dets, 5], torch.zeros((n_valid_dets, )))

    def test_batch_multiclass_nms(self, mocker):
        input_boxes, input_scores = generate_random_inputs(batch=3, n_boxes=20, n_classes=10)
        max_dets = 5

        # these numbers don't really make sense as nms outputs, but we don't really care, we only want to test
        # that outputs are combined correctly
        img_nms_ret = torch.rand(3, max_dets, 7)
        # scores
        img_nms_ret[..., 5] = torch.randint(0, 20, (3, max_dets), dtype=torch.float32)
        # input box indices
        img_nms_ret[..., 6] = torch.randint(0, 200, (3, max_dets), dtype=torch.float32)
        ret_valid_dets = Tensor([[5], [4], [3]])
        # each time the function is called, next value in the list returned
        images_ret = [(img_nms_ret[i], ret_valid_dets[i]) for i in range(3)]
        mock = mocker.patch('edgemdt_cl.pytorch.nms.nms_common._image_multiclass_nms',
                            Mock(side_effect=lambda *args, **kwargs: images_ret.pop(0)))

        res, n_valid = nms_common._batch_multiclass_nms(input_boxes,
                                                        input_scores,
                                                        score_threshold=0.1,
                                                        iou_threshold=0.6,
                                                        max_detections=5)

        # check each invocation
        for i, call_args in enumerate(mock.call_args_list):
            assert torch.equal(call_args.args[0], input_boxes[i]), i
            assert torch.equal(call_args.args[1], input_scores[i]), i
            assert call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5), i

        assert torch.equal(res, img_nms_ret)
        assert torch.equal(n_valid, ret_valid_dets)
