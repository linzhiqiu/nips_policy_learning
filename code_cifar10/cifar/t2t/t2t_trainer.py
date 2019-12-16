# coding=utf-8
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

"""tensor2tensor training wrapper with custom models and problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer
import tensorflow as tf
from off_policy_optimization.cifar.t2t import models  # pylint: disable=unused-import
from off_policy_optimization.cifar.t2t import problems  # pylint: disable=unused-import


def main(argv):
  t2t_trainer.main(argv)


if __name__ == "__main__":
  tf.app.run()
