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

"""Logs a trained linear policy to the BLBF dataset format txt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from off_policy_optimization.cifar.linear_logger import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path',
                    '/export/hda3/tmp/cifar_logger/train.tfrecords',
                    'location of tfrecords cifar file')

flags.DEFINE_string('checkpoint_path',
                    '/export/hda3/tmp/cifar_logger/linear_model.ckpt',
                    'location for saving checkpoint.')

flags.DEFINE_string('log_path',
                    '/export/hda3/tmp/cifar_logger/train_map_file_01121743.txt',
                    'location for saving blbf log data.')


flags.DEFINE_integer('epoch_size', 50000, 'num examples in epoch.')

flags.DEFINE_integer('batch_size', 100, 'batch size.')


def logsumexp(ns):
  max_ = np.max(ns)
  ds = ns - max_
  z = np.exp(ds).sum()
  return max_ + np.log(z)


def softmax(logits):
  return np.exp(logits - logsumexp(logits))


def policy(logits, tau=0.01):
  dims = logits.shape[0]
  probs = softmax(logits * tau)
  action = np.random.choice(dims, p=probs)
  return action, probs


def generate_string_token(idx=0,
                          action=1,
                          label=3,
                          loss=0,
                          probs=0.1 * np.ones(10)):
  """Parses the given row into the string format for logging."""
  token = '|id '
  token += str(int(idx)) + ' |'
  token += 'action ' + str(int(action)) + ' |'
  token += 'loss ' + str(loss) + ' |'
  token += 'fulllabel ' + str(int(label)) + ' |'
  token += 'prop ' + str(float(probs[action])) + ' |'
  token += 'probs '
  for p in probs:
    token += str(float(p)) + ' '
  token += '|\n'
  return token


def convert_batch_logs_to_txt(idx_batch, label_batch, logits_batch):
  """Converts a batch into the token format from the blbf dataset."""
  num_examples = len(idx_batch)
  t = ''
  errors = 0.
  for i in range(num_examples):
    logits = logits_batch[i]
    label = label_batch[i]
    idx = idx_batch[i]
    action, probs = policy(logits)
    loss = int(action != label)
    errors += loss
    token = generate_string_token(idx, action, label, loss, probs)
    t += token
  return t, errors / num_examples


def main(argv):
  del argv
  # Construct a copy of the graph
  graph = utils.build_graph(FLAGS.data_path, batch_size=100, train=False)
  init_op = tf.global_variables_initializer()
  saver = tf.train.Saver()
  sess = tf.Session()
  sess.run(init_op)
  saver.restore(sess, FLAGS.checkpoint_path)
  total_err = 0.
  total_err_logs = 0.
  full_token = ''
  batches_per_epoch = int(FLAGS.epoch_size / FLAGS.batch_size)
  for _ in range(batches_per_epoch):
    idx_batch, label_batch, logits_batch, err = sess.run(
        [graph['idx'], graph['label'], graph['logits'], graph['error']])
    total_err += err
    token, err_logs = convert_batch_logs_to_txt(idx_batch, label_batch,
                                                logits_batch)
    full_token += token
    total_err_logs += err_logs

  print(total_err / batches_per_epoch, total_err_logs / batches_per_epoch)

  with open(FLAGS.log_path, mode='w') as f:
    f.write(full_token)

if __name__ == '__main__':
  app.run(main)
