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
from typing import Callable

import torch

from edgemdt_cl.util.import_util import is_compatible

CUSTOM_LIB_NAME = 'edgemdt'
custom_lib = torch.library.Library(CUSTOM_LIB_NAME, "DEF")


def get_op_qualname(torch_op_name):
    """ Op qualified name """
    return CUSTOM_LIB_NAME + '::' + torch_op_name


def register_op(torch_op_name: str, schema: str, impl: Callable):
    """
    Register torch custom op under the custom library.

    Args:
        torch_op_name: op name to register.
        schema: schema for the custom op.
        impl: implementation of the custom op.

    Returns:
        Custom op qualified name.
    """
    torch_op_qualname = get_op_qualname(torch_op_name)

    custom_lib.define(schema)

    if is_compatible('torch>=2.2'):
        register_impl = torch.library.impl(torch_op_qualname, 'default')
    else:
        register_impl = torch.library.impl(custom_lib, torch_op_name)
    register_impl(impl)

    return torch_op_qualname
