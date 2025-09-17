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
from onnxruntime_extensions import onnx_op, PyCustomOpDef

from .nms_obb import _multiclass_nms_obb_impl
from .nms_obb_onnx import MULTICLASS_NMSOBB_ONNX_OP


@onnx_op(op_type=MULTICLASS_NMSOBB_ONNX_OP,
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
         outputs=[
             PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_int32, PyCustomOpDef.dt_float,
             PyCustomOpDef.dt_int32
         ],
         attrs={
             "score_threshold": PyCustomOpDef.dt_float,
             "iou_threshold": PyCustomOpDef.dt_float,
             "max_detections": PyCustomOpDef.dt_int64,
         })
def multiclass_nms_obb_ort(boxes, scores, angles, score_threshold, iou_threshold, max_detections):
    return _multiclass_nms_obb_impl(boxes, scores, angles, score_threshold, iou_threshold, max_detections)
