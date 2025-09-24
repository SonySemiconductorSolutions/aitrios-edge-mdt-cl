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

import abc

import tensorflow as tf

from edgemdt_cl.version import __version__


class CustomLayer(tf.keras.layers.Layer, abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_version: str = __version__

    def get_config(self):
        config = super().get_config()
        config['custom_version'] = self.custom_version
        return config

    @classmethod
    def from_config(cls, config):
        custom_version = config.pop('custom_version')
        layer = cls(**config)
        layer.custom_version = custom_version
        return layer
