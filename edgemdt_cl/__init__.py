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
import sys

# minimal requirements for dynamic validation in edgemdt_cl.{keras, pytorch}.__init__
# library names are the names that are used in import statement rather than pip package name, as a library can have
# multiple providing packages per arch, device etc.
if sys.version_info.major == 3 and sys.version_info.minor == 9:
    # onnxruntime<1.20 if python3.9
    required_libraries = {
        'tf': ['tensorflow>=2.14'],
        'torch': ['torch>=2.3', 'torchvision>=0.18'],
        'torch_ort': ['onnx>=1.14', 'onnxruntime>=1.15,<1.20', 'onnxruntime_extensions>=0.8.0'],
    }
else:
    required_libraries = {
        'tf': ['tensorflow>=2.14'],
        'torch': ['torch>=2.3', 'torchvision>=0.18'],
        'torch_ort': ['onnx>=1.14', 'onnxruntime>=1.15', 'onnxruntime_extensions>=0.8.0'],
    }

# pinned requirements of latest tested versions for extra_requires
if sys.version_info.major == 3 and sys.version_info.minor == 9:
    # onnxruntime<1.20 if python3.9
    pinned_pip_requirements = {
        'tf': ['tensorflow==2.15.*'],
        'torch': ['torch==2.6.*', 'torchvision==0.21.*'],
        'torch_ort': ['onnx==1.17.*', 'onnxruntime==1.19.*', 'onnxruntime_extensions==0.13.*']
else:
    pinned_pip_requirements = {
        'tf': ['tensorflow==2.15.*'],
        'torch': ['torch==2.6.*', 'torchvision==0.21.*'],
        'torch_ort': ['onnx==1.17.*', 'onnxruntime==1.21.*', 'onnxruntime_extensions==0.13.*']
    }
