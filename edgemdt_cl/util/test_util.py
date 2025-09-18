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
import subprocess as sp
import sys


def exec_in_clean_process(code: str, check: bool):
    # run in new process not contaminated be previous imports
    command = [sys.executable, '-c', code]
    p = sp.run(command, shell=False, check=False, capture_output=True, text=True)
    if check:
        assert p.returncode == 0, f'\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}'
    return p
