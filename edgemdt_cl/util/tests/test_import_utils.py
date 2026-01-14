# -----------------------------------------------------------------------------
# Copyright 2026 Sony Semiconductor Solutions, Inc. All rights reserved.
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
# flake8: noqa: E122
import pytest
from unittest.mock import MagicMock, patch

from edgemdt_cl.util.import_util import (is_compatible, validate_installed_libraries, RequirementError)


@pytest.mark.parametrize(("requirements", "mock_modules", "boolean_expected", "error_message_expected"), [
    (['torch'], {
        'torch': '2.3.0'
    }, True, None),
    (['torch>=2.3'], {
        'torch': '2.3.0'
    }, True, None),
    (['torch>=2.3', 'torchvision>=0.18'], {
        'torch': '2.3.0',
        'torchvision': '0.18.0'
    }, True, None),
    (['torch>=2.3', 'torchvision>=0.18'], {
        'torch': '2.3.0a0+dev0',
        'torchvision': '0.18.0.dev2+a0'
    }, True, None),
    (['torch>=2.3'], {}, False, "\nRequired library 'torch' is not installed."),
    (['torch>=2.3'], {
        'torch': '2.2.0'
    }, False, "\nRequired 'torch' version >=2.3, installed version 2.2.0."),
    (['torch>=2.3'], {
        'torch': '2.2.0a0+dev0'
    }, False, "\nRequired 'torch' version >=2.3, installed version 2.2.0a0+dev0."),
    (['torch>=2.3', 'torchvision>=0.18'], {
        'torch': '2.3.0'
    }, False, "\nRequired library 'torchvision' is not installed."),
    (['torch>=2.3', 'torchvision>=0.18'], {
        'torch': '2.2.0'
    }, False, "\nRequired 'torch' version >=2.3, installed version 2.2.0."
     "\nRequired library 'torchvision' is not installed."),
    (['torch>=2.3,<3.0'], {
        'torch': '2.5.0'
    }, True, None),
    (['torch>=2.3,<3.0'], {
        'torch': '2.5.0a0+dev0'
    }, True, None),
    (['torch>=2.3,<3.0'], {
        'torch': '3.0.0'
    }, False, "\nRequired 'torch' version <3.0,>=2.3, installed version 3.0.0."),
    (['torch>=2.3,<3.0'], {
        'torch': '2.2.0+dev1'
    }, False, "\nRequired 'torch' version <3.0,>=2.3, installed version 2.2.0+dev1."),
    (['tensorflow'], {
        'tensorflow': '2.15.0'
    }, True, None),
    (['tensorflow>=2.15'], {
        'tensorflow': '2.15.0'
    }, True, None),
    (['tensorflow>=2.15'], {
        'tensorflow': '2.15.0dev0'
    }, True, None),
    (['tensorflow>=2.15'], {}, False, "\nRequired library 'tensorflow' is not installed."),
    (['tensorflow>=2.15'], {
        'tensorflow': '2.10.0'
    }, False, "\nRequired 'tensorflow' version >=2.15, installed version 2.10.0."),
    (['tensorflow>=2.15'], {
        'tensorflow': '2.10.0dev0'
    }, False, "\nRequired 'tensorflow' version >=2.15, installed version 2.10.0dev0."),
    (['tensorflow>=2.14,<2.16'], {
        'tensorflow': '2.14.0'
    }, True, None),
    (['tensorflow>=2.14,<2.16'], {
        'tensorflow': '2.14.0dev0'
    }, True, None),
    (['tensorflow>=2.14,<2.16'], {
        'tensorflow': '1.10.0'
    }, False, "\nRequired 'tensorflow' version <2.16,>=2.14, installed version 1.10.0."),
    (['tensorflow>=2.14,<2.16'], {
        'tensorflow': '1.10.0dev1'
    }, False, "\nRequired 'tensorflow' version <2.16,>=2.14, installed version 1.10.0dev1."),
    (['onnx'], {
        'onnx': '1.17.0'
    }, True, None),
    (['onnx>=1.14'], {
        'onnx': '1.17.0'
    }, True, None),
    (['onnx>=1.14', 'onnxruntime>=1.15'], {
        'onnx': '1.17.0',
        'onnxruntime': '1.18.0'
    }, True, None),
    (['onnx>=1.14', 'onnxruntime>=1.15'], {
        'onnx': '1.17.0.dev0',
        'onnxruntime': '1.18.0dev2+a0'
    }, True, None),
    (['onnx>=1.14'], {}, False, "\nRequired library 'onnx' is not installed."),
    (['onnx>=1.14'], {
        'onnx': '1.0.0'
    }, False, "\nRequired 'onnx' version >=1.14, installed version 1.0.0."),
    (['onnx>=1.14'], {
        'onnx': '1.0.0.dev0'
    }, False, "\nRequired 'onnx' version >=1.14, installed version 1.0.0.dev0."),
    (['onnx>=1.14', 'onnxruntime>=1.15', 'onnxruntime_extensions>=0.8.0'], {
        'onnx': '1.17.0',
        'onnxruntime_extensions': '0.9.0'
    }, False, "\nRequired library 'onnxruntime' is not installed."),
    (['onnx>=1.14', 'onnxruntime>=1.15', 'onnxruntime_extensions>=0.8.0'], {
        'onnx': '1.0.0'
    }, False, "\nRequired 'onnx' version >=1.14, installed version 1.0.0."
     "\nRequired library 'onnxruntime' is not installed."
     "\nRequired library 'onnxruntime_extensions' is not installed."),
    (['onnx>=1.14,<1.18'], {
        'onnx': '1.17.0'
    }, True, None),
    (['onnx>=1.14,<1.18'], {
        'onnx': '1.17.0.dev0'
    }, True, None),
    (['onnx>=1.14,<1.18'], {
        'onnx': '1.3.0'
    }, False, "\nRequired 'onnx' version <1.18,>=1.14, installed version 1.3.0."),
    (['onnx>=1.14,<1.18'], {
        'onnx': '1.3.0.dev1'
    }, False, "\nRequired 'onnx' version <1.18,>=1.14, installed version 1.3.0.dev1."),
])
def test_validate_installed_libraries(requirements, mock_modules, boolean_expected, error_message_expected):

    def _create_mock_import_module(mock_modules):

        def mock_import_module(name):
            if name not in mock_modules:
                raise ImportError(f"No module named '{name}'")

            module = MagicMock()
            version = mock_modules[name]

            if version is not None:
                module.__version__ = version
            else:
                del module.__version__
            return module

        return mock_import_module

    mock_import_module = _create_mock_import_module(mock_modules)

    with patch('edgemdt_cl.util.import_util.importlib.import_module', side_effect=mock_import_module):
        # comparison using is_compatible()
        assert is_compatible(requirements) == boolean_expected

        # comparison using validate_installed_libraries()
        if error_message_expected is not None:
            with pytest.raises(RequirementError) as e:
                validate_installed_libraries(requirements)
            assert str(e.value) == error_message_expected
        else:
            validate_installed_libraries(requirements)
