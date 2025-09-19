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
from .nms import multiclass_nms, NMSResults, MulticlassNMS
from .nms_with_indices import multiclass_nms_with_indices, NMSWithIndicesResults, MulticlassNMSWithIndices

# trigger onnx op registration
from . import nms_onnx

__all__ = [
    'multiclass_nms', 'multiclass_nms_with_indices', 'NMSResults', 'NMSWithIndicesResults', 'MulticlassNMS',
    'MulticlassNMSWithIndices'
]
