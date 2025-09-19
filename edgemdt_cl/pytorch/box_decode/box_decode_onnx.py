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

from edgemdt_cl.pytorch.box_decode.box_decode import BOX_DECODE_TORCH_OP
from edgemdt_cl.pytorch.custom_lib import get_op_qualname

BOX_DECODE_ONNX_OP = "EdgeMDT::FasterRCNNBoxDecode"


@torch.onnx.symbolic_helper.parse_args('v', 'v', 'v', 'v')
def box_decode_onnx(g, rel_codes, anchors, scale_factors, clip_window):
    outputs = g.op(BOX_DECODE_ONNX_OP, rel_codes, anchors, scale_factors, clip_window, outputs=1)
    # Set output tensors shape and dtype
    outputs.setType(rel_codes.type())
    return outputs


torch.onnx.register_custom_op_symbolic(get_op_qualname(BOX_DECODE_TORCH_OP), box_decode_onnx, opset_version=1)
