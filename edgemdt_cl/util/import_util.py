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
import importlib
import warnings
from typing import List, Union
from packaging.requirements import Requirement
from packaging.version import parse


class RequirementError(Exception):
    pass


def validate_installed_libraries(requirements: List[str]):
    """
    Validate that all required libraries are installed and meet the version specifications.
    We import the required libraries and obtain the version from __version__, rather than looking at the installed
    pip package, since a single library can be provided by different pip packages per arch, device, etc.
    (for example 'import onnxruntime' is provided by both onnxruntime and onnxruntime-gpu packages).

    Args:
        requirements (list): a list of pip-style-like requirement strings with the package name being the library name
                             that is used in the import statement.

    Raises:
        RequirementError if any required library is not installed or doesn't meet the version specification
    """
    error = ''
    for req_str in requirements:
        req = Requirement(req_str)
        try:
            mod = importlib.import_module(req.name)
        except ImportError:
            error += f"\nRequired library '{req.name}' is not installed."
            continue

        if req.specifier:
            try:
                installed_ver = mod.__version__
            except AttributeError:
                warnings.warn(f"Failed to retrieve '{req.name}' version. Continuing without compatability check.")
                continue
            if parse(installed_ver) not in req.specifier:
                error += f"\nRequired '{req.name}' version {req.specifier}, installed version {installed_ver}."

    if error:
        raise RequirementError(error)


def is_compatible(requirements: Union[str, List]) -> bool:
    """
    Non-raising requirement(s) check
    Args:
        requirements (str, List): a pip-style-like requirement string with the package name being the library name that
                                  is used in the import statement, or a list of such requirement strings.

    Returns:
        (bool) whether requirement(s) are satisfied
    """
    requirements = [requirements] if isinstance(requirements, str) else requirements
    try:
        validate_installed_libraries(requirements)
    except RequirementError:
        return False
    return True
