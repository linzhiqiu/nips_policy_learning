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

"""Trains a simple linear policy to log the BLBF dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

from off_policy_optimization.cifar.linear_logger import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path',
                    '/export/hda3/tmp/cifar_logger/train.tfrecords',
                    'location of tfrecords cifar file')

flags.DEFINE_string('checkpoint_path',
                    '/export/hda3/tmp/cifar_logger/linear_model.ckpt',
                    'location for saving checkpoint.')

flags.DEFINE_integer('epoch_size', 50000, 'num examples in epoch.')

flags.DEFINE_integer('batch_size', 100, 'batch size.')


def main(argv):
  del argv
  graph = utils.build_graph(FLAGS.data_path, FLAGS.batch_size)
  init_op = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init_op)

  total_err = 0.
  batches_per_epoch = int(FLAGS.epoch_size / FLAGS.batch_size)
  for idx in range(batches_per_epoch * 10):
    _, err = sess.run([graph['train_op'], graph['error']])
    total_err += err
    if idx % batches_per_epoch == 0:
      print(total_err / batches_per_epoch)
      total_err = 0.

  saver = tf.train.Saver()
  save_path = saver.save(sess, FLAGS.checkpoint_path)
  print('Model saved in path: %s' % save_path)


if __name__ == '__main__':
  app.run(main)
