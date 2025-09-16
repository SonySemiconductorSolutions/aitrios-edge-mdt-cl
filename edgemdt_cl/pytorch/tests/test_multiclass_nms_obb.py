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
from unittest.mock import Mock

import pytest
import numpy as np
import torch
import onnxruntime as ort

from edgemdt_cl.pytorch import (multiclass_nms_obb, NMSOBBResults, MulticlassNMSOBB)
from edgemdt_cl.pytorch import load_custom_ops
from edgemdt_cl.pytorch.nms_obb.nms_obb_common import LABELS, INDICES, SCORES, ANGLES
from edgemdt_cl.pytorch.tests.test_nms_obb_common import generate_random_inputs_obb
from edgemdt_cl.pytorch.tests.util import load_and_validate_onnx_model, check_tensor
from edgemdt_cl.util.test_util import exec_in_clean_process


class TestMultiClassNMSOBB:

    def _batch_multiclass_nms_obb_mock(self, batch, n_dets, n_classes=20):
        ret = torch.rand(batch, n_dets, 8)
        ret[..., LABELS] = torch.randint(n_classes, size=(batch, n_dets), dtype=torch.float32)    # labels
        ret[..., INDICES] = torch.randint(n_dets * n_classes, size=(batch, n_dets),
                                          dtype=torch.float32)    # input box indices
        n_valid = torch.randint(n_dets + 1, size=(3, 1), dtype=torch.float32)
        return Mock(return_value=(ret, n_valid))

    def test_torch_op(self, mocker):
        mock = mocker.patch(f'edgemdt_cl.pytorch.nms_obb.nms_obb._batch_multiclass_nms_obb',
                            self._batch_multiclass_nms_obb_mock(batch=3, n_dets=5))
        boxes, scores, angles = generate_random_inputs_obb(batch=3, n_boxes=10, n_classes=5)
        ret = torch.ops.edgemdt.multiclass_nms_obb(boxes, scores, angles, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert torch.equal(mock.call_args.args[2], angles)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5)

        assert torch.equal(ret[0], mock.return_value[0][:, :, :4])
        assert ret[0].dtype == torch.float32
        assert torch.equal(ret[1], mock.return_value[0][:, :, SCORES])
        assert ret[1].dtype == torch.float32
        assert torch.equal(ret[2], mock.return_value[0][:, :, LABELS])
        assert ret[2].dtype == torch.int64
        assert torch.equal(ret[3], mock.return_value[0][:, :, ANGLES])
        assert ret[3].dtype == torch.float32
        assert torch.equal(ret[4], mock.return_value[1])
        assert ret[4].dtype == torch.int64
        assert len(ret) == 5

    def test_torch_op_wrapper(self, mocker):
        mock = mocker.patch(f'edgemdt_cl.pytorch.nms_obb.nms_obb._batch_multiclass_nms_obb',
                            self._batch_multiclass_nms_obb_mock(batch=3, n_dets=5))
        boxes, scores, angles = generate_random_inputs_obb(batch=3, n_boxes=20, n_classes=10)
        ret = multiclass_nms_obb(boxes, scores, angles, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert torch.equal(mock.call_args.args[2], angles)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5)

        ref_ret = torch.ops.edgemdt.multiclass_nms_obb(boxes, scores, angles, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert isinstance(ret, NMSOBBResults)

        assert torch.equal(ret.boxes, ref_ret[0])
        assert ret.boxes.dtype == torch.float32
        assert torch.equal(ret.scores, ref_ret[1])
        assert ret.scores.dtype == torch.float32
        assert torch.equal(ret.labels, ref_ret[2])
        assert ret.labels.dtype == torch.int64
        assert torch.equal(ret.angles, ref_ret[3])
        assert ret.angles.dtype == torch.float32
        assert torch.equal(ret.n_valid, ref_ret[4])
        assert ret.n_valid.dtype == torch.int64

    @pytest.mark.parametrize('cuda', [True, False])
    def test_full_op_sanity(self, cuda):
        if cuda and not torch.cuda.is_available():
            pytest.skip('cuda is not available')
        boxes, scores, angles = generate_random_inputs_obb(batch=3, n_boxes=20, n_classes=10)
        multiclass_nms_obb(boxes, scores, angles, score_threshold=0.1, iou_threshold=0.6, max_detections=5)

    def test_empty_tensors(self):
        # empty inputs
        ret = multiclass_nms_obb(torch.rand(1, 0, 4), torch.rand(1, 0, 10), torch.rand(1, 0, 1), 0.55, 0.6, 50)
        assert ret.n_valid[0] == 0 and ret.boxes.size(1) == 50

    def test_score_threshold(self):
        # score_threshold is too high
        ret = multiclass_nms_obb(torch.rand(1, 100, 4), torch.rand(1, 100, 20) / 2, torch.rand(1, 100, 1), 1.0, 0.3, 50)
        assert ret.n_valid[0] == 0 and ret.boxes.size(1) == 50

    def test_iou_threshold(self):
        # iou_threshold is too low
        ret = multiclass_nms_obb(torch.rand(1, 100, 4), torch.rand(1, 100, 20), torch.rand(1, 100, 1), 0.3, 0.0, 50)
        assert ret.n_valid[0] == 0 and ret.boxes.size(1) == 50

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    def test_onnx_export(self, dynamic_batch, tmpdir_factory):
        score_thresh = 0.1
        iou_thresh = 0.6
        n_boxes = 10
        n_classes = 5
        max_dets = 7

        onnx_model = MulticlassNMSOBB(score_thresh, iou_thresh, max_dets)

        path = str(tmpdir_factory.mktemp('nms_obb').join(f'nms_obb.onnx'))
        self._export_onnx(onnx_model, n_boxes, n_classes, path, dynamic_batch=dynamic_batch)

        onnx_model = load_and_validate_onnx_model(path, exp_opset=1)

        nms_node = list(onnx_model.graph.node)[0]
        assert nms_node.domain == 'EdgeMDT'
        assert nms_node.op_type == 'MultiClassNMSOBB'
        attrs = sorted(nms_node.attribute, key=lambda a: a.name)
        assert attrs[0].name == 'iou_threshold'
        np.isclose(attrs[0].f, iou_thresh)
        assert attrs[1].name == 'max_detections'
        assert attrs[1].i == max_dets
        assert attrs[2].name == 'score_threshold'
        np.isclose(attrs[2].f, score_thresh)
        assert len(nms_node.input) == 3
        assert len(nms_node.output) == 5

        check_tensor(onnx_model.graph.input[0], [10, 4], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch) # boxes
        check_tensor(onnx_model.graph.input[1], [10, 5], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch) # scores
        check_tensor(onnx_model.graph.input[2], [10, 1], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch) # angles
        # test shape inference that is defined as part of onnx op
        check_tensor(onnx_model.graph.output[0], [max_dets, 4], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch) # boxes
        check_tensor(onnx_model.graph.output[1], [max_dets], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch) # scores
        check_tensor(onnx_model.graph.output[2], [max_dets], torch.onnx.TensorProtoDataType.INT32, dynamic_batch) # labels
        check_tensor(onnx_model.graph.output[3], [max_dets], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch) # angles
        check_tensor(onnx_model.graph.output[4], [1], torch.onnx.TensorProtoDataType.INT32, dynamic_batch) # n_valid

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    def test_ort(self, dynamic_batch, tmpdir_factory):
        model = MulticlassNMSOBB(score_threshold=0.5, iou_threshold=0.3, max_detections=1000)
        n_boxes = 500
        n_classes = 20
        path = str(tmpdir_factory.mktemp('nms_obb').join(f'nms_obb.onnx'))
        self._export_onnx(model, n_boxes, n_classes, path, dynamic_batch)

        batch = 5 if dynamic_batch else 1
        boxes, scores, angles = generate_random_inputs_obb(batch=batch, n_boxes=n_boxes, n_classes=n_classes, seed=42)
        torch_res = model(boxes, scores, angles)
        so = load_custom_ops()
        session = ort.InferenceSession(path, sess_options=so)
        ort_res = session.run(output_names=None, input_feed={'boxes': boxes.numpy(), 'scores': scores.numpy(), 'angles': angles.numpy()})
        # this is just a sanity test on random data
        for i in range(len(torch_res)):
            assert np.allclose(torch_res[i], ort_res[i])
        # run in a new process
        code = f"""
import onnxruntime as ort
import numpy as np
from edgemdt_cl.pytorch import load_custom_ops
so = ort.SessionOptions()
so = load_custom_ops(so)
session = ort.InferenceSession('{path}', so)
boxes = np.random.rand({batch}, {n_boxes}, 4).astype(np.float32)
scores = np.random.rand({batch}, {n_boxes}, {n_classes}).astype(np.float32)
angles = np.random.rand({batch}, {n_boxes}, 1).astype(np.float32)
ort_res = session.run(output_names=None, input_feed={{'boxes': boxes, 'scores': scores, 'angles': angles}})
        """
        exec_in_clean_process(code, check=True)

    def _export_onnx(self, nms_model, n_boxes, n_classes, path, dynamic_batch: bool):
        input_names = ['boxes', 'scores', 'angles']
        output_names = ['det_boxes', 'det_scores', 'det_labels', 'det_angles', 'valid_dets']

        kwargs = {'dynamic_axes': {k: {0: 'batch'} for k in input_names + output_names}} if dynamic_batch else {}
        torch.onnx.export(nms_model,
                          args=(torch.ones(1, n_boxes, 4), torch.ones(1, n_boxes, n_classes), torch.ones(1, n_boxes, 1)),
                          f=path,
                          input_names=input_names,
                          output_names=output_names,
                          **kwargs)


class TestMultiClassNMSOBBE2E:

    def setup(self):
        # Batch size is assumed to be 1.
        self.boxes = torch.Tensor([[10, 10, 4, 3],
                                   [10.5, 10, 4.5, 2.5],
                                   [20, 20, 1, 5],
                                   [20.5, 20, 1.5, 5],
                                   [30, 30, 1, 5],
                                   [30, 31, 1, 6],
                                   [40, 40, 2, 2]]).unsqueeze(0)    # yapf: disable
        self.scores = torch.Tensor([[0.2, 0.1, 0.25],
                                    [0.2, 0.1, 0.3],
                                    [0.3, 0.2, 0.05],
                                    [0.1, 0.4, 0.05],
                                    [0.05, 0.1, 0.5],
                                    [0.1, 0.15, 0.55],
                                    [0.15, 0.1, 0.05]]).unsqueeze(0)    # yapf: disable
        self.angles = torch.Tensor([[0.78],
                                    [0.785],
                                    [1.5],
                                    [1.3],
                                    [0.1],
                                    [2.3],
                                    [-0.5]]).unsqueeze(0)    # yapf: disable
        self.score_threshold = 0.1
        self.iou_threshold = 0.6
        self.max_detections = 10

        self.exp_boxes = torch.Tensor([[30.0, 31.0,  1.0,  6.0],
                                       [30.0, 30.0,  1.0,  5.0],
                                       [20.5, 20.0,  1.5,  5.0],
                                       [10.5, 10.0,  4.5,  2.5],
                                       [20.0, 20.0,  1.0,  5.0],
                                       [40.0, 40.0,  2.0,  2.0]]).unsqueeze(0)  # yapf: disable
        self.exp_scores = torch.Tensor([0.55, 0.5, 0.4, 0.3, 0.3, 0.15]).unsqueeze(0)
        self.exp_labels = torch.Tensor([2, 2, 1, 2, 0, 0]).unsqueeze(0)
        self.exp_angles = torch.Tensor([2.3, 0.1, 1.3, 0.785, 1.5, -0.5]).unsqueeze(0)
        self.exp_valid_dets = 6

    def test_multiclass_nms_obb(self):
        self.setup()
        ret = multiclass_nms_obb(self.boxes, 
                                 self.scores, 
                                 self.angles, 
                                 score_threshold=self.score_threshold,
                                 iou_threshold=self.iou_threshold,
                                 max_detections=self.max_detections)
        
        assert isinstance(ret, NMSOBBResults)
        
        # check for boxes
        assert ret.boxes.shape == (1, self.max_detections, 4)
        assert ret.boxes.dtype == torch.float32
        assert torch.equal(ret.boxes[:, :self.exp_valid_dets, :], self.exp_boxes[:self.exp_valid_dets])
        assert torch.all(ret.boxes[:, self.exp_valid_dets:, :] == 0)
        # check for scores
        assert ret.scores.shape == (1, self.max_detections)
        assert ret.scores.dtype == torch.float32
        assert torch.equal(ret.scores[:, :self.exp_valid_dets], self.exp_scores[:self.exp_valid_dets])
        assert torch.all(ret.scores[:, self.exp_valid_dets:] == 0)
        # check for labels
        assert ret.labels.shape == (1, self.max_detections)
        assert ret.labels.dtype == torch.int64
        assert torch.equal(ret.labels[:, :self.exp_valid_dets], self.exp_labels[:self.exp_valid_dets])
        assert torch.all(ret.labels[:, self.exp_valid_dets:] == 0)
        # check for angles
        assert ret.angles.shape == (1, self.max_detections)
        assert ret.angles.dtype == torch.float32
        assert torch.equal(ret.angles[:, :self.exp_valid_dets], self.exp_angles[:self.exp_valid_dets])
        assert torch.all(ret.angles[:, self.exp_valid_dets:] == 0)
        # check for n_valid
        assert ret.n_valid.shape == (1, 1)
        assert ret.n_valid.dtype == torch.int64
        assert ret.n_valid == torch.Tensor([self.exp_valid_dets])

    def test_class_multiclass_nms_obb(self):
        self.setup()
        nms_obb = MulticlassNMSOBB(score_threshold=self.score_threshold,
                                   iou_threshold=self.iou_threshold,
                                   max_detections=self.max_detections)
        ret = nms_obb(self.boxes, self.scores, self.angles)
        
        assert isinstance(ret, NMSOBBResults)
        
        # check for boxes
        assert ret.boxes.shape == (1, self.max_detections, 4)
        assert ret.boxes.dtype == torch.float32
        assert torch.equal(ret.boxes[:, :self.exp_valid_dets, :], self.exp_boxes[:self.exp_valid_dets])
        assert torch.all(ret.boxes[:, self.exp_valid_dets:, :] == 0)
        # check for scores
        assert ret.scores.shape == (1, self.max_detections)
        assert ret.scores.dtype == torch.float32
        assert torch.equal(ret.scores[:, :self.exp_valid_dets], self.exp_scores[:self.exp_valid_dets])
        assert torch.all(ret.scores[:, self.exp_valid_dets:] == 0)
        # check for labels
        assert ret.labels.shape == (1, self.max_detections)
        assert ret.labels.dtype == torch.int64
        assert torch.equal(ret.labels[:, :self.exp_valid_dets], self.exp_labels[:self.exp_valid_dets])
        assert torch.all(ret.labels[:, self.exp_valid_dets:] == 0)
        # check for angles
        assert ret.angles.shape == (1, self.max_detections)
        assert ret.angles.dtype == torch.float32
        assert torch.equal(ret.angles[:, :self.exp_valid_dets], self.exp_angles[:self.exp_valid_dets])
        assert torch.all(ret.angles[:, self.exp_valid_dets:] == 0)
        # check for n_valid
        assert ret.n_valid.shape == (1, 1)
        assert ret.n_valid.dtype == torch.int64
        assert ret.n_valid == torch.Tensor([self.exp_valid_dets])
