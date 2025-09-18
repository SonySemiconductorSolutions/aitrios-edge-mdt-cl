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

from typing import Type
import tensorflow as tf


def custom_layers_scope(*args: dict):
    """
    Scope context manager that can be used to deserialize Keras models containing custom layers

    If the model contains custom layers only from this package:
    ```
    from edgemdt_cl.keras import custom_layers_scope
    with custom_layers_scope():
        tf.keras.models.load_model(path)
    ```
    If the model contains additional custom layers from other sources, there are two ways:
    1. Pass a list of dictionaries {layer_name: layer_object} as *args.
        ```
        with custom_layers_scope({'Op1': Op1, 'Op2': Op2}, {'Op3': Op3}):
            tf.keras.models.load_model(path)
        ```
    2. Combined with other scopes based on tf.keras.utils.custom_object_scope:
        ```
        with custom_layers_scope(), another_scope():
            tf.keras.models.load_model(path)

        # or:

        with custom_layers_scope():
            with another_scope():
                tf.keras.models.load_model(path)
        ```
    Args:
        *args: a list of dictionaries for other custom layers

    Returns:
        Scope context manager
    """
    return tf.keras.utils.custom_object_scope(*args + (_custom_objects, ))


_custom_objects = {}


def register_layer(kls: Type) -> Type:
    """ decorator to automatically add custom layer to custom objects dict """
    _custom_objects[kls.__name__] = kls
    return kls
