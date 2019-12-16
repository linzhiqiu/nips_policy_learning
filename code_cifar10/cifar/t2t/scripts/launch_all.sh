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

export PATH_PREFIX=off_policy_optimization/cifar/t2t/scripts
#${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh
${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/l2.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/so.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/so1.sh
#
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/combo.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/combo12.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/combo_02_95_02.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/combo23.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/combo13.sh
#
#
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b05.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b1.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b15.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b2.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b4.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b6.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b8.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/pg_b10.sh
#
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/so_rt1.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/so_rt2.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/so_rt3.sh
# ${PATH_PREFIX}/launch_expt_local.sh ${PATH_PREFIX}/cifar_base.sh ${PATH_PREFIX}/loss_configs/so_rt4.sh
