# Copyright 2019 The Off Policy Optimization Authors.
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

#!/bin/bash

export BASE_DIR=/tmp/tfexamples/CIFAR-10-BLBF
PATH_PREFIX=off_policy_optimization/cifar/t2t/scripts
${PATH_PREFIX}/generate_blbf_data.sh ${PATH_PREFIX}/data_configs/cifar_linear_logger.sh
# ${PATH_PREFIX}/generate_blbf_data.sh ${PATH_PREFIX}/data_configs/cifar_blbf_half.sh
# ${PATH_PREFIX}/generate_blbf_data.sh ${PATH_PREFIX}/data_configs/cifar_blbf1.sh
# ${PATH_PREFIX}/generate_blbf_data.sh ${PATH_PREFIX}/data_configs/cifar_blbf2.sh
# ${PATH_PREFIX}/generate_blbf_data.sh ${PATH_PREFIX}/data_configs/cifar_blbf3.sh
# ${PATH_PREFIX}/generate_blbf_data.sh ${PATH_PREFIX}/data_configs/cifar_blbf5.sh
