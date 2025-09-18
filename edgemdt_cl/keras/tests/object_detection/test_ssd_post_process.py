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

from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf
import pytest

from edgemdt_cl.keras.object_detection import SSDPostProcess, ScoreConverter
from edgemdt_cl.keras.tests.common import CustomOpTesterBase


@pytest.fixture
def scale_factors():
    return 1, 2, 3, 4


@pytest.fixture
def img_size():
    return 100, 200


scores_func = {
    ScoreConverter.LINEAR: lambda scores: scores,
    ScoreConverter.SIGMOID: lambda scores: 1 / (1 + np.exp(-scores)),
    ScoreConverter.SOFTMAX: lambda scores: np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True),
}


class TestSSDPostProcess(CustomOpTesterBase):

    @pytest.mark.parametrize('score_conv, remove_bg',
                             [(s, True) for s in list(ScoreConverter)] + [(s.value, False)
                                                                          for s in ScoreConverter])    # as enum or str
    def test_flow(self, mocker, score_conv, remove_bg, scale_factors, img_size):
        batch_size = 3
        n_boxes = 5
        n_labels = 10
        score_thresh = 0.21
        iou_thesh = 0.31
        max_detections = 123
        anchors = np.random.uniform(0, 1, size=(n_boxes, 4)).astype(np.float32)

        # mock box decode inference to return pre-set boxes
        boxes = np.random.rand(batch_size, n_boxes, 4)
        from edgemdt_cl.keras.object_detection import FasterRCNNBoxDecode
        bd_call = mocker.patch.object(FasterRCNNBoxDecode, 'call', Mock(return_value=boxes))

        # mock nms op
        nms = mocker.patch('tensorflow.image.combined_non_max_suppression')

        post_process = SSDPostProcess(anchors=anchors,
                                      scale_factors=scale_factors,
                                      clip_size=img_size,
                                      score_converter=score_conv,
                                      score_threshold=score_thresh,
                                      iou_threshold=iou_thesh,
                                      max_detections=max_detections,
                                      remove_background=remove_bg)

        rel_codes = np.random.uniform(0, 1, size=(batch_size, n_boxes, 4)).astype(np.float32)
        scores = np.random.uniform(0, 1, size=(batch_size, n_boxes, n_labels)).astype(np.float32)

        out = post_process([rel_codes, scores])

        # verify box decode params and input are set correctly
        assert np.array_equal(post_process._box_decode.scale_factors, np.asarray(scale_factors, dtype=np.float32))
        assert np.array_equal(post_process._box_decode.anchors, anchors)
        assert post_process._box_decode.clip_window == (0, 0, *img_size)
        assert np.array_equal(bd_call.call_args[0][0].numpy(), rel_codes)

        # verify nms params and inputs are set correctly
        exp_input_boxes = np.expand_dims(boxes, axis=-2)
        assert np.array_equal(nms.call_args.args[0].numpy(), exp_input_boxes)
        exp_input_scores = scores_func[score_conv](scores)
        if remove_bg:
            exp_input_scores = exp_input_scores[..., 1:]
        assert np.allclose(nms.call_args.args[1].numpy(), exp_input_scores)
        assert nms.call_args.kwargs == {
            'max_output_size_per_class': max_detections,
            'max_total_size': max_detections,
            'iou_threshold': iou_thesh,
            'score_threshold': score_thresh,
            'pad_per_class': False,
            'clip_boxes': False
        }
        assert out == nms.return_value

    def test_sanity_nms(self, mocker, scale_factors, img_size):
        n_boxes = 5
        # non-overlapping boxes
        boxes = np.array([[[0, .05, .1, .15],
                           [.2, .25, .3, .35],
                           [.4, .45, .5, .55],
                           [.6, .65, .7, .75],
                           [.8, .85, .9, .95]]]).astype(np.float32)    # yapf: disable

        # mock box decode inference to return pre-set boxes
        from edgemdt_cl.keras.object_detection import FasterRCNNBoxDecode
        mocker.patch.object(FasterRCNNBoxDecode, 'call', Mock(return_value=boxes))

        # each valid score appears once to prevent ambiguity in order
        scores = np.array([[[.1, .21, .3],
                            [.23, .1, .1],
                            [.1, .25, .22],
                            [.1, .1, .1],
                            [0, 0, .5]]]).astype(np.float32)    # yapf: disable
        score_threshold = .2
        max_detections = 10

        exp_score = np.array([[.5, .3, .25, .23, .22, .21]]).astype(np.float32)
        exp_box_ind = [4, 0, 2, 1, 2, 0]
        exp_label = np.array([[2, 2, 1, 0, 2, 1]]).astype(np.float32)
        exp_n_valid = min(len(exp_box_ind), max_detections)

        anchors = np.random.uniform(0, 1, size=(n_boxes, 4)).astype(np.float32)
        pp = SSDPostProcess(anchors=anchors,
                            scale_factors=scale_factors,
                            clip_size=img_size,
                            score_converter=ScoreConverter.LINEAR,
                            score_threshold=score_threshold,
                            iou_threshold=0.1,
                            max_detections=max_detections)

        rel_codes = np.random.uniform(0, 1, size=(1, n_boxes, 4)).astype(np.float32)
        selected_boxes, scores, labels, n_valid = pp([rel_codes, scores])
        assert selected_boxes.shape == (1, max_detections, 4)
        assert labels.shape == scores.shape == (1, max_detections)
        assert n_valid.numpy().item() == exp_n_valid
        assert np.array_equal(scores.numpy()[:, :exp_n_valid], exp_score)
        assert np.array_equal(labels.numpy()[:, :exp_n_valid], exp_label)
        for i, ind in enumerate(exp_box_ind):
            assert np.array_equal(selected_boxes[0, i, :], boxes[0, ind, :])

    @pytest.mark.parametrize(
        'scale_factors, clip_size, n_boxes, n_labels, max_detections, remove_bg',
        [
            ((1, 2, 3, 4),
             (10., 20.), 200, 10, 100, True),    # int factors, float size, n_boxes * n_labels > max_detections
            ((1.1, 2.1, 3.1, 4.1),
             (0, 1), 15, 10, 200, False),    # float factors, int size, n_boxes * n_labels < max_detections
        ])
    def test_full_op_model(self, tmp_path, scale_factors, clip_size, n_boxes, n_labels, max_detections, remove_bg,
                           save_format_ext):
        batch_size = 10
        score_thresh = 0.5
        iou_thresh = 0.6
        score_conv = ScoreConverter.SIGMOID
        anchors = np.random.uniform(0, 1, size=(n_boxes, 4)).astype(np.float32)
        with patch('edgemdt_cl.keras.base_custom_layer.__version__', 'foo.bar'):
            post_process = SSDPostProcess(anchors=anchors,
                                          scale_factors=scale_factors,
                                          clip_size=clip_size,
                                          score_converter=score_conv,
                                          score_threshold=score_thresh,
                                          iou_threshold=iou_thresh,
                                          max_detections=max_detections,
                                          remove_background=remove_bg,
                                          name='pp')

        orig_model = self._build_model(post_process, n_boxes, n_labels)
        path = tmp_path / f'model{save_format_ext}'
        orig_model.save(path)

        # check that the model can be loaded from a clean process (not contaminated by previous imports)
        self._test_clean_load_model_with_custom_objects(path)

        from edgemdt_cl.keras import custom_layers_scope
        with custom_layers_scope():
            model = tf.keras.models.load_model(path)

        assert model.layers[-1].custom_version == 'foo.bar'

        cfg = model.layers[-1].get_config()
        assert np.array_equal(cfg['anchors'], anchors)
        assert tuple(cfg['scale_factors']) == scale_factors
        assert tuple(cfg['clip_size']) == clip_size
        assert cfg['score_converter'] == score_conv
        assert cfg['score_threshold'] == score_thresh
        assert cfg['iou_threshold'] == iou_thresh
        assert cfg['max_detections'] == max_detections
        assert cfg['remove_background'] == remove_bg
        assert cfg['name'] == 'pp'

        rel_codes = np.random.uniform(0, 1, size=(batch_size, n_boxes, 4)).astype(np.float32)
        scores = np.random.randn(batch_size, n_boxes, n_labels).astype(np.float32)
        out_boxes, out_scores, out_labels, out_n_valid = model([rel_codes, scores])
        assert out_boxes.shape == (batch_size, max_detections, 4)
        assert out_scores.shape == out_labels.shape == (batch_size, max_detections)
        assert out_n_valid.shape == (batch_size, )
        valid_score_cnt = np.sum(np.count_nonzero(scores_func[score_conv](scores) > score_thresh, axis=-1), axis=1)
        exp_n_valid = np.minimum(valid_score_cnt, max_detections)
        assert exp_n_valid.shape == (batch_size, )
        assert np.array_equal(out_n_valid.numpy(), exp_n_valid)

    def _build_model(self, post_process_op, n_boxes, n_labels):
        rel_codes = tf.keras.layers.Input(shape=(n_boxes, 4))
        scores = tf.keras.layers.Input(shape=(n_boxes, n_labels))
        out = post_process_op([rel_codes, scores])
        model = tf.keras.Model(inputs=[rel_codes, scores], outputs=out)
        return model
