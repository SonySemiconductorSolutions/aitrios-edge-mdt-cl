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
import torch
from onnxruntime_extensions import onnx_op, PyCustomOpDef

from .box_decode import _faster_rcnn_box_decode_impl
from .box_decode_onnx import BOX_DECODE_ONNX_OP


@onnx_op(op_type=BOX_DECODE_ONNX_OP,
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
         outputs=[PyCustomOpDef.dt_float])
def box_decode_ort(rel_codes, anchors, scale_factors, clip_window):
    return _faster_rcnn_box_decode_impl(torch.as_tensor(rel_codes), torch.as_tensor(anchors),
                                        torch.as_tensor(scale_factors), torch.as_tensor(clip_window))
