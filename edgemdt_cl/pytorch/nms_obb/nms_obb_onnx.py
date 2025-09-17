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
import torch

from edgemdt_cl.pytorch.custom_lib import get_op_qualname
from .nms_obb import MULTICLASS_NMSOBB_TORCH_OP

MULTICLASS_NMSOBB_ONNX_OP = "EdgeMDT::MultiClassNMSOBB"


@torch.onnx.symbolic_helper.parse_args('v', 'v', 'v', 'f', 'f', 'i')
def multiclass_nms_obb_onnx(g, boxes, scores, angles, score_threshold, iou_threshold, max_detections):
    outputs = g.op(MULTICLASS_NMSOBB_ONNX_OP,
                   boxes,
                   scores,
                   angles,
                   score_threshold_f=score_threshold,
                   iou_threshold_f=iou_threshold,
                   max_detections_i=max_detections,
                   outputs=5)
    # Set output tensors shape and dtype
    # Based on examples in https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/
    # training/ortmodule/_custom_op_symbolic_registry.py (see cross_entropy_loss)
    # This is a hack to set output type that is different from input type. Apparently it cannot be set directly
    output_int_type = g.op("Cast", boxes, to_i=torch.onnx.TensorProtoDataType.INT32).type()
    batch = torch.onnx.symbolic_helper._get_tensor_dim_size(boxes, 0)
    outputs[0].setType(boxes.type().with_sizes([batch, max_detections, 4]))
    outputs[1].setType(scores.type().with_sizes([batch, max_detections]))
    outputs[2].setType(output_int_type.with_sizes([batch, max_detections]))
    outputs[3].setType(angles.type().with_sizes([batch, max_detections]))
    outputs[4].setType(output_int_type.with_sizes([batch, 1]))
    return outputs


torch.onnx.register_custom_op_symbolic(get_op_qualname(MULTICLASS_NMSOBB_TORCH_OP),
                                       multiclass_nms_obb_onnx,
                                       opset_version=1)
