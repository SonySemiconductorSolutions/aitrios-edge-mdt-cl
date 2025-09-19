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
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import pytest

from edgemdt_cl.keras.object_detection import FasterRCNNBoxDecode
from edgemdt_cl.keras.tests.common import CustomOpTesterBase


class TestBoxDecode(CustomOpTesterBase):

    @pytest.mark.parametrize('scale_factors, clip_window, tf_anchors', [([1, 2, 3, 4], [-1.5, 1.5, 10.1, 20.1], False),
                                                                        ([1.1, 2.2, 3.3, 4.4], [0, 1, 2, 3], True)])
    def test_load_custom(self, scale_factors, clip_window, tmp_path, tf_anchors, save_format_ext):
        shape = (10, 4)
        path = tmp_path / f'model{save_format_ext}'
        anchors = self._generate_random_anchors(n_anchors=10, seed=1)
        if tf_anchors:
            anchors = tf.constant(anchors)

        with patch('edgemdt_cl.keras.base_custom_layer.__version__', 'foo.bar'):
            orig_model = self._build_model(anchors, scale_factors, clip_window)
        orig_model.save(path)

        # check that the model can be loaded from a clean process (not contaminated by previous imports)
        self._test_clean_load_model_with_custom_objects(path)

        from edgemdt_cl.keras import custom_layers_scope
        with custom_layers_scope():
            model = tf.keras.models.load_model(path)

        cfg = model.layers[-1].get_config()
        assert np.array_equal(cfg['anchors'], anchors)
        assert np.array_equal(cfg['scale_factors'], np.asarray(scale_factors).astype(np.float32))
        assert list(cfg['clip_window']) == clip_window
        # is inferable
        model(np.random.uniform(size=(1, *shape)).astype(np.float32))
        # version
        assert model.layers[-1].custom_version == 'foo.bar'

    def test_zero_offsets(self):
        n_boxes = 100
        anchors = self._generate_random_anchors(n_boxes, seed=1)
        model = self._build_model(anchors, (1, 2, 3, 4), (0, 0, 1, 1))
        out = model(np.zeros((2, n_boxes, 4)).astype(np.float32))
        assert np.allclose(out, anchors)

    @pytest.mark.parametrize('scale_factors', [(1., 1., 1., 1.), (1, 2, 3, 4), (1.1, 2.2, 0.5, 3.3)])
    def test_box_decode(self, scale_factors):
        n_boxes = 100
        anchors = self._generate_random_anchors(n_boxes, img_size=(100, 200), seed=1)

        v0, v1, v2, v3 = .5, 1., .2, 1.2
        offsets = np.empty((2, n_boxes, 4), dtype=np.float32)
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
        model = self._build_model(anchors, scale_factors, clip_window=(-1000, -1000, 1000, 1000))
        boxes = model(offsets)
        boxes_hw = boxes[..., 2:] - boxes[..., :2]
        anchors_hw = anchors[..., 2:] - anchors[..., :2]
        assert np.allclose(boxes_hw[..., 0] / anchors_hw[..., 0], np.exp(v2))
        assert np.allclose(boxes_hw[..., 1] / anchors_hw[..., 1], np.exp(v3))
        boxes_center = boxes[..., :2] + 0.5 * boxes_hw
        anchors_center = anchors[..., :2] + 0.5 * anchors_hw
        t = (boxes_center - anchors_center) / anchors_hw
        assert np.allclose(t[..., 0], v0, atol=1e-5)
        assert np.allclose(t[..., 1], v1, atol=1e-5)

    @pytest.mark.parametrize('clip_window, normalize', [((-4, 1, 90, 110), False), ((-.04, .01, .9, 1.1), True)])
    def test_clipping(self, clip_window, normalize):
        scale_factors = (1, 2, 3, 4)
        n_boxes = 3
        anchors = self._generate_random_anchors(n_anchors=n_boxes, seed=1)
        mul = 0.01 if normalize else 1
        # (2, n_boxes, 4)
        boxes = mul * np.array([
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
        ]).astype(np.float32)

        rel_codes = self._encode_offsets(boxes, anchors, scale_factors=scale_factors)
        model = self._build_model(anchors, scale_factors=scale_factors, clip_window=clip_window)
        out = model(rel_codes)
        exp_boxes = mul * np.array([
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
        ]).astype(np.float32)
        assert np.allclose(out, exp_boxes, atol=1e-6)

    @staticmethod
    def _generate_random_boxes(n_batches, n_boxes, seed=None):
        if seed:
            np.random.seed(seed)
        boxes = np.empty((n_batches, n_boxes, 4))
        boxes[..., :2] = np.random.uniform(low=0, high=.9, size=(n_batches, n_boxes, 2))
        boxes[..., 2:] = np.random.uniform(low=boxes[..., :2], high=1., size=(n_batches, n_boxes, 2))
        return boxes.astype(np.float32)

    @classmethod
    def _generate_random_anchors(cls, n_anchors, img_size=None, seed=None):
        anchors = cls._generate_random_boxes(1, n_anchors, seed)[0]
        if img_size:
            anchors = anchors * np.array(img_size + img_size, dtype=np.float32)
        return anchors

    @staticmethod
    def _encode_offsets(boxes, anchors, scale_factors):
        anchors_hw = anchors[..., 2:] - anchors[..., :2]
        boxes_hw = boxes[..., 2:] - boxes[..., :2]
        boxes_center = boxes[..., :2] + boxes_hw / 2
        anchors_center = anchors[..., :2] + anchors_hw / 2
        thw = np.log(boxes_hw / anchors_hw)
        tyx = (boxes_center - anchors_center) / anchors_hw
        t = np.concatenate([tyx, thw], axis=-1)
        return t * np.asarray(scale_factors)

    @staticmethod
    def _build_model(anchors, scale_factors, clip_window):
        box_decode = FasterRCNNBoxDecode(anchors=anchors, scale_factors=scale_factors, clip_window=clip_window)
        x = tf.keras.layers.Input(anchors.shape)
        model = tf.keras.Model(x, box_decode(x))
        return model
