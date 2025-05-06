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
from typing import Optional

import numpy as np
import onnx.helper
import pytest
import torch
import onnxruntime as ort
from edgemdt_cl.util.test_util import exec_in_clean_process

from edgemdt_cl.pytorch import FasterRCNNBoxDecode, load_custom_ops
from edgemdt_cl.pytorch.tests.util import load_and_validate_onnx_model


class TestBoxDecode:

    def test_zero_offsets(self):
        n_boxes = 100
        anchors = self._generate_random_anchors(n_boxes, seed=1)
        model = FasterRCNNBoxDecode(anchors=anchors, scale_factors=(1, 2, 3, 4), clip_window=(0, 0, 1, 1))
        out = model(torch.zeros((2, n_boxes, 4), dtype=torch.float32))
        assert torch.allclose(out, anchors)

    @pytest.mark.parametrize('scale_factors', [(1., 1., 1., 1.), (1, 2, 3, 4), (1.1, 2.2, 0.5, 3.3)])
    @pytest.mark.parametrize('cuda', [False, True])
    def test_box_decode(self, scale_factors, cuda):
        if cuda and not torch.cuda.is_available():
            pytest.skip('cuda is not available')

        n_boxes = 100
        anchors = self._generate_random_anchors(n_boxes, img_size=(100, 200), seed=1)

        v0, v1, v2, v3 = .5, 1., .2, 1.2
        offsets = torch.empty((2, n_boxes, 4), dtype=torch.float32)
        # we define encoded offsets that will yield boxes such that:
        # np.log(boxes_h / anchors_h) = v2
        # np.log(boxes_w / anchors_w) = v3
        # (boxes_center_y - anchors_center_y) / anchors_h = v0
        # (boxes_center_x - anchors_center_x) / anchors_w = v1
        offsets[:, :, 0] = v0 * scale_factors[0]
        offsets[:, :, 1] = v1 * scale_factors[1]
        offsets[:, :, 2] = v2 * scale_factors[2]
        offsets[:, :, 3] = v3 * scale_factors[3]

        # disable clipping
        model = FasterRCNNBoxDecode(anchors, scale_factors, clip_window=(-1000, -1000, 1000, 1000))
        if cuda:
            model = model.cuda()
            offsets = offsets.cuda()
        boxes = model(offsets)

        boxes = boxes.cpu()
        boxes_hw = boxes[..., 2:] - boxes[..., :2]
        anchors_hw = anchors[..., 2:] - anchors[..., :2]
        assert torch.allclose(boxes_hw[..., 0] / anchors_hw[..., 0], torch.exp(torch.as_tensor(v2)))
        assert torch.allclose(boxes_hw[..., 1] / anchors_hw[..., 1], torch.exp(torch.as_tensor(v3)))
        boxes_center = boxes[..., :2] + 0.5 * boxes_hw
        anchors_center = anchors[..., :2] + 0.5 * anchors_hw
        t = (boxes_center - anchors_center) / anchors_hw
        assert torch.allclose(t[..., 0], torch.as_tensor(v0), atol=1e-5)
        assert torch.allclose(t[..., 1], torch.as_tensor(v1), atol=1e-5)

    @pytest.mark.parametrize('clip_window, normalize', [((-4, 1, 90, 110), False), ((-.04, .01, .9, 1.1), True)])
    def test_clipping(self, clip_window, normalize):
        scale_factors = (1, 2, 3, 4)
        n_boxes = 3
        anchors = self._generate_random_anchors(n_anchors=n_boxes, seed=1)
        mul = 0.01 if normalize else 1
        # (2, n_boxes, 4)
        boxes = mul * torch.as_tensor(
            [
                [
                    [-5, 5, 1, 12],    # clip y_min
                    [85, -4, 90, 2],    # clip x_min
                    [85, 95, 95, 100]    # clip y_max
                ],
                [
                    [0, 85, 2, 115],    # clip x_max
                    [-10, 115, -5, 120],    # y_min, y_max < 0, x_min, x_max > x_size
                    [95, -10, 100, -5]    # y_min, y_max > y_size, x_min, x_max < 0
                ]
            ],
            dtype=torch.float32)

        rel_codes = self._encode_offsets(boxes, anchors, scale_factors=scale_factors)
        model = FasterRCNNBoxDecode(anchors, scale_factors=scale_factors, clip_window=clip_window)
        out = model(rel_codes)
        exp_boxes = mul * torch.as_tensor(
            [
                [
                    [-4, 5, 1, 12],    # clip y_min
                    [85, 1, 90, 2],    # clip x_min
                    [85, 95, 90, 100]    # clip y_max
                ],
                [
                    [0, 85, 2, 110],    # clip x_max
                    [-4, 110, -4, 110],    # y_min, y_max < 0, x_min, x_max > x_size
                    [90, 1, 90, 1]    # y_min, y_max > y_size, x_min, x_max < 0
                ]
            ],
            dtype=torch.float32)
        assert torch.allclose(out, exp_boxes, atol=1e-6)

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    @pytest.mark.parametrize('scale_factors, clip_window', [
        [(1, 2, 3, 4), (0.1, 0.2, 0.9, 1.2)],
        [(1.1, 2.2, 3.3, 0.5), (10, 20, 30, 40)],
    ])
    def test_onnx_export(self, dynamic_batch, scale_factors, clip_window, tmp_path):
        n_boxes = 1000
        anchors = self._generate_random_anchors(n_anchors=n_boxes, seed=1)
        model = FasterRCNNBoxDecode(anchors, scale_factors=scale_factors, clip_window=clip_window)
        path = str(tmp_path / 'box_decode.onnx')
        self._export_onnx(model, n_boxes, path, dynamic_batch=dynamic_batch)

        onnx_model = load_and_validate_onnx_model(path, exp_opset=1)

        [box_decode_node] = list(onnx_model.graph.node)
        assert box_decode_node.domain == 'EdgeMDT'
        assert box_decode_node.op_type == 'FasterRCNNBoxDecode'
        assert len(box_decode_node.input) == 4
        assert len(box_decode_node.output) == 1
        # sanity check that we extracted the input nodes correctly
        anchors_input, scale_factors_input, clip_window_input = list(onnx_model.graph.initializer)
        assert box_decode_node.input[1] == anchors_input.name
        assert box_decode_node.input[2] == scale_factors_input.name
        assert box_decode_node.input[3] == clip_window_input.name

        def check_input(t, exp_tensor):
            assert tuple(t.dims) == exp_tensor.shape
            assert np.allclose(onnx.numpy_helper.to_array(t), exp_tensor)

        check_input(anchors_input, anchors)
        check_input(scale_factors_input, np.array(scale_factors))
        check_input(clip_window_input, np.array(clip_window))

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    @pytest.mark.parametrize('scale_factors, clip_window', [
        [(1, 2, 3, 4), (0.1, 0.2, 0.9, 1.2)],
        [(1.1, 2.2, 3.3, 0.5), (10, 20, 30, 40)],
    ])
    def test_ort(self, dynamic_batch, scale_factors, clip_window, tmp_path):
        n_boxes = 1000
        anchors = self._generate_random_anchors(n_anchors=n_boxes, seed=1)
        model = FasterRCNNBoxDecode(anchors, scale_factors=scale_factors, clip_window=clip_window)
        path = str(tmp_path / 'box_decode.onnx')
        self._export_onnx(model, n_boxes, path, dynamic_batch=dynamic_batch)

        batch = 5 if dynamic_batch else 1
        boxes = self._generate_random_boxes(batch, n_boxes, seed=1)
        rel_codes = self._encode_offsets(boxes, anchors, scale_factors)

        torch_res = model(rel_codes)
        so = load_custom_ops()

        session = ort.InferenceSession(path, sess_options=so)
        ort_res = session.run(output_names=None, input_feed={'rel_codes': rel_codes.numpy()})
        assert np.allclose(torch_res, ort_res[0])

        # run in a new process
        code = f"""
import onnxruntime as ort
import numpy as np
from edgemdt_cl.pytorch import load_custom_ops
so = ort.SessionOptions()
so = load_custom_ops(so)
session = ort.InferenceSession('{path}', so)
rel_codes = np.random.rand({batch}, {n_boxes}, 4).astype(np.float32)
ort_res = session.run(output_names=None, input_feed={{'rel_codes': rel_codes}})
assert ort_res[0].max() and ort_res[0].max() > ort_res[0].min()
        """
        exec_in_clean_process(code, check=True)

    @staticmethod
    def _generate_random_boxes(n_batches, n_boxes, seed=None):
        if seed:
            np.random.seed(seed)
        boxes = np.empty((n_batches, n_boxes, 4))
        boxes[..., :2] = np.random.uniform(low=0, high=.9, size=(n_batches, n_boxes, 2))
        boxes[..., 2:] = np.random.uniform(low=boxes[..., :2], high=1., size=(n_batches, n_boxes, 2))
        return torch.as_tensor(boxes, dtype=torch.float32)

    @classmethod
    def _generate_random_anchors(cls, n_anchors, img_size: Optional[tuple] = None, seed=None):
        anchors = cls._generate_random_boxes(1, n_anchors, seed)[0]
        if img_size:
            anchors = anchors * torch.tensor(img_size + img_size, dtype=torch.float32)
        return anchors

    @staticmethod
    def _encode_offsets(boxes, anchors, scale_factors):
        anchors_hw = anchors[..., 2:] - anchors[..., :2]
        boxes_hw = boxes[..., 2:] - boxes[..., :2]
        boxes_center = boxes[..., :2] + boxes_hw / 2
        anchors_center = anchors[..., :2] + anchors_hw / 2
        thw = torch.log(boxes_hw / anchors_hw)
        tyx = (boxes_center - anchors_center) / anchors_hw
        t = torch.concat([tyx, thw], dim=-1)
        return t * torch.as_tensor(scale_factors)

    def _export_onnx(self, model, n_boxes, path, dynamic_batch: bool):
        input_names = ['rel_codes']
        output_names = ['decoded']
        kwargs = {'dynamic_axes': {k: {0: 'batch'} for k in input_names + output_names}} if dynamic_batch else {}
        torch.onnx.export(model,
                          args=(torch.ones((1, n_boxes, 4))),
                          f=path,
                          input_names=input_names,
                          output_names=output_names,
                          **kwargs)
