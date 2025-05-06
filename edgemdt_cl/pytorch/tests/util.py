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
import onnx


def load_and_validate_onnx_model(path, exp_opset):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model, full_check=True)
    opset_info = list(onnx_model.opset_import)[1]
    assert opset_info.domain == 'EdgeMDT' and opset_info.version == exp_opset
    return onnx_model


def check_tensor(onnx_tensor, exp_shape, exp_type, dynamic_batch: bool):
    tensor_type = onnx_tensor.type.tensor_type
    shape = [d.dim_value if d.dim_value else d.dim_param for d in tensor_type.shape.dim]
    exp_shape = ['batch' if dynamic_batch else 1] + exp_shape
    assert shape == exp_shape
    assert tensor_type.elem_type == exp_type
