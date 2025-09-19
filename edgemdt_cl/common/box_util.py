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
from typing import Tuple


def corners_to_centroids(y_min, x_min, y_max, x_max) -> Tuple:
    """ Converts corners coordinates into centroid coordinates """
    height = y_max - y_min
    width = x_max - x_min
    y_center = y_min + 0.5 * height
    x_center = x_min + 0.5 * width
    return y_center, x_center, height, width


def centroids_to_corners(y_center, x_center, height, width) -> Tuple:
    """ Converts centroid coordinates into corners coordinates """
    y_min = y_center - 0.5 * height
    x_min = x_center - 0.5 * width
    y_max = y_min + height
    x_max = x_min + width
    return y_min, x_min, y_max, x_max
