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
from typing import Optional, TYPE_CHECKING

from edgemdt_cl.util.import_util import validate_installed_libraries
from edgemdt_cl import required_libraries
from edgemdt_cl.pytorch.custom_layer import CustomLayer

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = [
    'multiclass_nms', 'NMSResults', 'multiclass_nms_with_indices', 'NMSWithIndicesResults', 'FasterRCNNBoxDecode',
    'load_custom_ops', 'MulticlassNMS', 'MulticlassNMSWithIndices', 'MulticlassNMSOBB', 'multiclass_nms_obb',
    'NMSOBBResults', 'CustomLayer'
]

validate_installed_libraries(required_libraries['torch'])
from edgemdt_cl.pytorch.nms import (    # noqa: E402
    multiclass_nms, NMSResults, multiclass_nms_with_indices, NMSWithIndicesResults, MulticlassNMS,
    MulticlassNMSWithIndices)
from edgemdt_cl.pytorch.nms_obb import multiclass_nms_obb, NMSOBBResults, MulticlassNMSOBB    # noqa: E402
from edgemdt_cl.pytorch.box_decode import FasterRCNNBoxDecode    # noqa: E402


def load_custom_ops(ort_session_ops: Optional['ort.SessionOptions'] = None) -> 'ort.SessionOptions':
    """
    Registers the custom ops implementation for onnxruntime, and sets up the SessionOptions object for onnxruntime
    session.

    Args:
        ort_session_ops: SessionOptions object to register the custom ops library on. If None, creates a new object.

    Returns:
        SessionOptions object with registered custom ops.

    Example:
        ```
        import onnxruntime as ort
        from edgemdt_cl.pytorch import load_custom_ops

        so = load_custom_ops()
        session = ort.InferenceSession(model_path, sess_options=so)
        session.run(...)
        ```
        You can also pass your own SessionOptions object upon which to register the custom ops
        ```
        load_custom_ops(ort_session_options=so)
        ```
    """
    validate_installed_libraries(required_libraries['torch_ort'])

    # trigger onnxruntime op registration
    from .nms import nms_ort
    from .nms_obb import nms_obb_ort
    from .box_decode import box_decode_ort

    from onnxruntime_extensions import get_library_path
    from onnxruntime import SessionOptions
    ort_session_ops = ort_session_ops or SessionOptions()
    ort_session_ops.register_custom_ops_library(get_library_path())
    return ort_session_ops
