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

if [[ "$#" -lt 1 ]]; then
    echo "You must enter at least one variable setting script."
    echo "Sample usage: ./launch.sh cifar_base.sh loss_l2.sh"
    echo "To override a subset of variables, simply add them to another variable script."
    exit -1
fi

echo "Including variable setting scripts:"

for script in "$@"
do
  echo "$script"
  source "$script"
done


echo "running the trainer locally with eval..."

python -m off_policy_optimization.cifar.t2t.t2t_trainer \
    --logtostderr \
    --schedule evaluate \
    --data_dir $DATA_DIR \
    --problem $PROBLEM \
    --model $MODEL \
    --hparams_set $HPARAMS \
    --output_dir ${OUTPUT_DIR} \
    --eval_steps 78 \
    --eval_delay_secs 0 \
