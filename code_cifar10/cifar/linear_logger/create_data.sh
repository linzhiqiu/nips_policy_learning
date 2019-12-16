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
export TRAIN_PATH=${BASE_DIR}/train.tfrecords
export CHECKPOINT_PATH=${BASE_DIR}/linear_model.ckpt
export LOG_PATH=${BASE_DIR}/train_map_file_linear0.txt

python -m off_policy_optimization.cifar.linear_logger.generate_tfrecords --data_dir ${BASE_DIR}

python -m off_policy_optimization.cifar.linear_logger.train --data_path ${TRAIN_PATH} \
--checkpoint_path ${CHECKPOINT_PATH}

python -m off_policy_optimization.cifar.linear_logger.linear_logger \
--data_path ${TRAIN_PATH} \
--checkpoint_path ${CHECKPOINT_PATH} \
--log_path ${LOG_PATH}
