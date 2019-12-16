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

"""Read CIFAR-10 data and writes TFRecords with partial feedback/bandit data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tarfile
from absl import app
from absl import flags
import numpy as np
from six.moves import cPickle as pickle
from six.moves import xrange
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/export/hda3/tmp/cifar_logger',
                    'data directory')


def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
  # file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict


def convert_to_tfrecord(input_files, output_file, pc=0.49):
  """Samples actions as indicated by pc and writes the result to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    example_idx = 0
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)
      assert num_entries_in_batch == 10000
      for i in range(num_entries_in_batch):
        # Create the log data
        if np.random.uniform() < pc:
          action = labels[i]
        else:
          action = np.random.randint(10)
        if action == labels[i]:
          propensity = pc + (1 - pc) * 0.1
          reward = 1.0
        else:
          propensity = (1 - pc) * 0.1
          reward = 0.0
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'example_idx': _int64_feature(example_idx),
                    'action': _int64_feature(action),
                    'reward': _float_feature(reward),
                    'propensity': _float_feature(propensity),
                    'image': _bytes_feature(data[i].tobytes()),
                    'label': _int64_feature(labels[i])
                }))
        record_writer.write(example.SerializeToString())
        example_idx += 1


def main(argv):
  del argv
  data_dir = FLAGS.data_dir
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(data_dir)
  file_names = _get_file_names()
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
  for mode, files in file_names.items():
    input_files = [os.path.join(input_dir, f) for f in files]
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_files, output_file)
  print('Done!')


if __name__ == '__main__':
  app.run(main)
