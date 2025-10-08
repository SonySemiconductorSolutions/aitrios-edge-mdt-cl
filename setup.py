# -------------------------------------------------------------------------------
# (c) Copyright 2023 Sony Semiconductor Solutions, Inc. All rights reserved.
#
#      This software, in source or object form (the "Software"), is the
#      property of Sony Semiconductor Solutions Inc. (the "Company") and/or its
#      licensors, which have all right, title and interest therein, You
#      may use the Software only in accordance with the terms of written
#      license agreement between you and the Company (the "License").
#      Except as expressly stated in the License, the Company grants no
#      licenses by implication, estoppel, or otherwise. If you are not
#      aware of or do not agree to the License terms, you may not use,
#      copy or modify the Software. You may use the source code of the
#      Software only for your internal purposes and may not distribute the
#      source code of the Software, any part thereof, or any derivative work
#      thereof, to any third party, except pursuant to the Company's prior
#      written consent.
#      The Software is the confidential information of the Company.
# -------------------------------------------------------------------------------
"""
Created on 7/9/23

@author: irenab
"""
from setuptools import setup

from edgemdt_cl import pinned_pip_requirements

extras_require = {
    'torch': pinned_pip_requirements['torch'] + pinned_pip_requirements['torch_ort'],
    'tf': pinned_pip_requirements['tf'],
}

setup(extras_require=extras_require, 
      python_requires='>=3.9')
