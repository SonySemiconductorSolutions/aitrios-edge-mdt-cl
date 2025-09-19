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

from edgemdt_cl.util.test_util import exec_in_clean_process


class CustomOpTesterBase:

    @staticmethod
    def _test_clean_load_model_with_custom_objects(path):
        cmd = f"""
import tensorflow as tf
from edgemdt_cl.keras import custom_layers_scope
with custom_layers_scope():
    tf.keras.models.load_model('{path}')
"""
        exec_in_clean_process(cmd, check=True)

    @staticmethod
    def _test_load_model(path):
        cmd = f"import tensorflow as tf;" \
              f"tf.keras.models.load_model('{path}')"
        exec_in_clean_process(cmd, check=True)
