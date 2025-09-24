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

# minimal requirements for dynamic validation in edgemdt_cl.{keras, pytorch}.__init__
# library names are the names that are used in import statement rather than pip package name, as a library can have
# multiple providing packages per arch, device etc.
required_libraries = {
    'tf': ['tensorflow>=2.10'],
    'torch': ['torch>=2.0', 'torchvision>=0.15'],
    'torch_ort': ['onnx', 'onnxruntime', 'onnxruntime_extensions>=0.8.0'],
}

# pinned requirements of latest tested versions for extra_requires
pinned_pip_requirements = {
    'tf': ['tensorflow==2.15.*'],
    'torch': ['torch==2.5.*', 'torchvision==0.20.*'],
    'torch_ort': ['onnx==1.17.*', 'onnxruntime==1.21.*', 'onnxruntime_extensions==0.13.*']
}
