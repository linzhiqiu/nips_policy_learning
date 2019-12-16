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

export PROBLEM=image_cifar10_blbf_linear_logger0
# export PROBLEM=image_cifar10_blbf1
# export PROBLEM=image_cifar10_blbf2
# export PROBLEM=image_cifar10_blbf3
# export PROBLEM=image_cifar10_blbf5
# export PROBLEM=image_cifar10_blbf_half

export DATA_DIR=/tmp/tfexamples/CIFAR-10-BLBF/${PROBLEM}
export OUTPUT_DIR=/tmp/cifar_blbf/${PROBLEM}

export MTYPE=resnet
export MODEL=${MTYPE}_partial_feedback
export HPARAMS=${MTYPE}_baseline
