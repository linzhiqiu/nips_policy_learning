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

"""Linear logger utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3

FLAGS = flags.FLAGS


def parser(serialized_example):
  """parse function for tf.data pipeline."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'example_idx': tf.FixedLenFeature([], tf.int64),
      })
  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([DEPTH * HEIGHT * WIDTH])

  # Reshape from [depth * height * width] to [depth, height, width].
  image = tf.cast(
      tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
      tf.float32)
  label = tf.cast(features['label'], tf.int32)
  example_idx = tf.cast(features['example_idx'], tf.int32)

  return image, label, example_idx


def build_graph(data_path, batch_size=100, train=True):
  """Builds the graph for logging policy."""
  dset = tf.data.TFRecordDataset(data_path).repeat()
  dset = dset.map(parser).batch(batch_size)
  dit = dset.make_one_shot_iterator()
  output = dit.get_next()
  img, lbl, idx = output
  img_reshaped = tf.reshape(img, [-1, HEIGHT * WIDTH * DEPTH])
  logits = tf.layers.dense(img_reshaped, 10)
  error = tf.reduce_mean(1.0 - tf.cast(
      tf.equal(tf.argmax(logits, 1), tf.cast(lbl, dtype=tf.int64)), tf.float32))
  loss = tf.losses.sparse_softmax_cross_entropy(
      labels=tf.squeeze(lbl), logits=logits)
  loss_tensor = tf.reduce_mean(loss)
  train_op = None
  if train:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss_tensor)
  graph = {
      'train_op': train_op,
      'logits': logits,
      'error': error,
      'idx': idx,
      'label': lbl
  }
  return graph
