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
from unittest.mock import Mock

import pytest
import numpy as np
import torch
import onnxruntime as ort

from edgemdt_cl.pytorch import (multiclass_nms, multiclass_nms_with_indices, NMSResults, NMSWithIndicesResults,
                                        MulticlassNMS, MulticlassNMSWithIndices)
from edgemdt_cl.pytorch import load_custom_ops
from edgemdt_cl.pytorch.nms.nms_common import LABELS, INDICES, SCORES
from edgemdt_cl.pytorch.tests.test_nms_common import generate_random_inputs
from edgemdt_cl.pytorch.tests.util import load_and_validate_onnx_model, check_tensor
from edgemdt_cl.util.test_util import exec_in_clean_process


class TestMultiClassNMS:

    def _batch_multiclass_nms_mock(self, batch, n_dets, n_classes=20):
        ret = torch.rand(batch, n_dets, 7)
        ret[..., LABELS] = torch.randint(n_classes, size=(batch, n_dets), dtype=torch.float32)    # labels
        ret[..., INDICES] = torch.randint(n_dets * n_classes, size=(batch, n_dets),
                                          dtype=torch.float32)    # input box indices
        n_valid = torch.randint(n_dets + 1, size=(3, 1), dtype=torch.float32)
        return Mock(return_value=(ret, n_valid))

    @pytest.mark.parametrize('op, patch_pkg', [(torch.ops.edgemdt.multiclass_nms, 'nms'),
                                               (torch.ops.edgemdt.multiclass_nms_with_indices, 'nms_with_indices')])
    def test_torch_op(self, mocker, op, patch_pkg):
        mock = mocker.patch(f'edgemdt_cl.pytorch.nms.{patch_pkg}._batch_multiclass_nms',
                            self._batch_multiclass_nms_mock(batch=3, n_dets=5))
        boxes, scores = generate_random_inputs(batch=3, n_boxes=10, n_classes=5)
        ret = op(boxes, scores, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5.)
        assert torch.equal(ret[0], mock.return_value[0][:, :, :4])
        assert ret[0].dtype == torch.float32
        assert torch.equal(ret[1], mock.return_value[0][:, :, SCORES])
        assert ret[1].dtype == torch.float32
        assert torch.equal(ret[2], mock.return_value[0][:, :, LABELS])
        assert ret[2].dtype == torch.int64
        if op == torch.ops.edgemdt.multiclass_nms_with_indices:
            assert torch.equal(ret[3], mock.return_value[0][:, :, INDICES])
            assert ret[3].dtype == torch.int64
            assert torch.equal(ret[4], mock.return_value[1])
            assert ret[4].dtype == torch.int64
            assert len(ret) == 5
        elif op == torch.ops.edgemdt.multiclass_nms:
            assert torch.equal(ret[3], mock.return_value[1])
            assert ret[3].dtype == torch.int64
            assert len(ret) == 4
        else:
            raise ValueError(op)

    @pytest.mark.parametrize('op, res_cls, torch_op, patch_pkg',
                             [(multiclass_nms, NMSResults, torch.ops.edgemdt.multiclass_nms, 'nms'),
                              (multiclass_nms_with_indices, NMSWithIndicesResults,
                               torch.ops.edgemdt.multiclass_nms_with_indices, 'nms_with_indices')])
    def test_torch_op_wrapper(self, mocker, op, res_cls, torch_op, patch_pkg):
        mock = mocker.patch(f'edgemdt_cl.pytorch.nms.{patch_pkg}._batch_multiclass_nms',
                            self._batch_multiclass_nms_mock(batch=3, n_dets=5))
        boxes, scores = generate_random_inputs(batch=3, n_boxes=20, n_classes=10)
        ret = op(boxes, scores, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5)

        ref_ret = torch_op(boxes, scores, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert isinstance(ret, res_cls)
        assert torch.equal(ret.boxes, ref_ret[0])
        assert ret.boxes.dtype == torch.float32
        assert torch.equal(ret.scores, ref_ret[1])
        assert ret.scores.dtype == torch.float32
        assert torch.equal(ret.labels, ref_ret[2])
        assert ret.labels.dtype == torch.int64
        if op == multiclass_nms:
            assert torch.equal(ret.n_valid, ref_ret[3])
            assert ret.n_valid.dtype == torch.int64
        elif op == multiclass_nms_with_indices:
            assert torch.equal(ret.indices, ref_ret[3])
            assert ret.indices.dtype == torch.int64
            assert torch.equal(ret.n_valid, ref_ret[4])
            assert ret.n_valid.dtype == torch.int64

    @pytest.mark.parametrize('op', [multiclass_nms, multiclass_nms_with_indices])
    @pytest.mark.parametrize('cuda', [True, False])
    def test_full_op_sanity(self, op, cuda):
        if cuda and not torch.cuda.is_available():
            pytest.skip('cuda is not available')
        boxes, scores = generate_random_inputs(batch=3, n_boxes=20, n_classes=10)
        op(boxes, scores, score_threshold=0.1, iou_threshold=0.6, max_detections=5)

    @pytest.mark.parametrize('op', [multiclass_nms, multiclass_nms_with_indices])
    def test_empty_tensors(self, op):
        # empty inputs
        ret = op(torch.rand(1, 0, 4), torch.rand(1, 0, 10), 0.55, 0.6, 50)
        assert ret.n_valid[0] == 0 and ret.boxes.size(1) == 50
        # no valid scores
        ret = op(torch.rand(1, 100, 4), torch.rand(1, 100, 20) / 2, 0.55, 0.6, 50)
        assert ret.n_valid[0] == 0 and ret.boxes.size(1) == 50

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    @pytest.mark.parametrize('with_indices', [True, False])
    def test_onnx_export(self, dynamic_batch, tmpdir_factory, with_indices):
        score_thresh = 0.1
        iou_thresh = 0.6
        n_boxes = 10
        n_classes = 5
        max_dets = 7

        nms_class = MulticlassNMSWithIndices if with_indices else MulticlassNMS
        onnx_model = nms_class(score_thresh, iou_thresh, max_dets)

        path = str(tmpdir_factory.mktemp('nms').join(f'nms{with_indices}.onnx'))
        self._export_onnx(onnx_model, n_boxes, n_classes, path, dynamic_batch=dynamic_batch, with_indices=with_indices)

        onnx_model = load_and_validate_onnx_model(path, exp_opset=1)

        nms_node = list(onnx_model.graph.node)[0]
        assert nms_node.domain == 'EdgeMdt'
        assert nms_node.op_type == ('MultiClassNMSWithIndices' if with_indices else 'MultiClassNMS')
        attrs = sorted(nms_node.attribute, key=lambda a: a.name)
        assert attrs[0].name == 'iou_threshold'
        np.isclose(attrs[0].f, iou_thresh)
        assert attrs[1].name == 'max_detections'
        assert attrs[1].i == max_dets
        assert attrs[2].name == 'score_threshold'
        np.isclose(attrs[2].f, score_thresh)
        assert len(nms_node.input) == 2
        assert len(nms_node.output) == 4 + int(with_indices)

        check_tensor(onnx_model.graph.input[0], [10, 4], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch)
        check_tensor(onnx_model.graph.input[1], [10, 5], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch)
        # test shape inference that is defined as part of onnx op
        check_tensor(onnx_model.graph.output[0], [max_dets, 4], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch)
        check_tensor(onnx_model.graph.output[1], [max_dets], torch.onnx.TensorProtoDataType.FLOAT, dynamic_batch)
        check_tensor(onnx_model.graph.output[2], [max_dets], torch.onnx.TensorProtoDataType.INT32, dynamic_batch)
        if with_indices:
            check_tensor(onnx_model.graph.output[3], [max_dets], torch.onnx.TensorProtoDataType.INT32, dynamic_batch)
            check_tensor(onnx_model.graph.output[4], [1], torch.onnx.TensorProtoDataType.INT32, dynamic_batch)
        else:
            check_tensor(onnx_model.graph.output[3], [1], torch.onnx.TensorProtoDataType.INT32, dynamic_batch)

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    @pytest.mark.parametrize('with_indices', [True, False])
    def test_ort(self, dynamic_batch, tmpdir_factory, with_indices):
        nms_class = MulticlassNMSWithIndices if with_indices else MulticlassNMS
        model = nms_class(score_threshold=0.5, iou_threshold=0.3, max_detections=1000)
        n_boxes = 500
        n_classes = 20
        path = str(tmpdir_factory.mktemp('nms').join(f'nms{with_indices}.onnx'))
        self._export_onnx(model, n_boxes, n_classes, path, dynamic_batch, with_indices=with_indices)

        batch = 5 if dynamic_batch else 1
        boxes, scores = generate_random_inputs(batch=batch, n_boxes=n_boxes, n_classes=n_classes, seed=42)
        torch_res = model(boxes, scores)
        so = load_custom_ops()
        session = ort.InferenceSession(path, sess_options=so)
        ort_res = session.run(output_names=None, input_feed={'boxes': boxes.numpy(), 'scores': scores.numpy()})
        # this is just a sanity test on random data
        for i in range(len(torch_res)):
            assert np.allclose(torch_res[i], ort_res[i]), i
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
ort_res = session.run(output_names=None, input_feed={{'boxes': boxes, 'scores': scores}})
        """
        exec_in_clean_process(code, check=True)

    def _export_onnx(self, nms_model, n_boxes, n_classes, path, dynamic_batch: bool, with_indices: bool):
        input_names = ['boxes', 'scores']
        output_names = ['det_boxes', 'det_scores', 'det_labels', 'valid_dets']
        if with_indices:
            output_names.insert(3, 'indices')
        kwargs = {'dynamic_axes': {k: {0: 'batch'} for k in input_names + output_names}} if dynamic_batch else {}
        torch.onnx.export(nms_model,
                          args=(torch.ones(1, n_boxes, 4), torch.ones(1, n_boxes, n_classes)),
                          f=path,
                          input_names=input_names,
                          output_names=output_names,
                          **kwargs)
