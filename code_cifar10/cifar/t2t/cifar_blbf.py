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

"""CIFAR Batch Learning from Bandit Feedback (BLBF) datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
import six
from six.moves import cPickle
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry
import tensorflow as tf
from off_policy_optimization.cifar.t2t import blbf_utils

# URLs and filenames for CIFAR data.
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_PREFIX = "cifar-10-batches-py/"
_CIFAR10_TRAIN_FILES = [
    "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
    "data_batch_5"
]
_CIFAR10_TEST_FILES = ["test_batch"]
_CIFAR10_IMAGE_SIZE = 32
_CIFAR10_BLBF_PATH = "/export/hda3/tfexamples/CIFAR-10-BLBF"
_CIFAR_BLBF_MAP_FILES = [
    "train_map_blbf1.txt", "train_map_blbf2.txt", "train_map_blbf3.txt",
    "train_map_blbf5.txt"
]
_LINEAR_LOGGER_PATH = _CIFAR10_BLBF_PATH
_LINEAR_MAP_FILES = ["train_map_file_linear0.txt"]


def _get_cifar(directory, url):
  """Download and extract CIFAR to directory unless it is there."""
  filename = os.path.basename(url)
  path = generator_utils.maybe_download(directory, filename, url)
  tarfile.open(path, "r:gz").extractall(directory)


def _parse_blbf_file(fname, fraction=1.0):
  """Parses the blbf text file."""
  f = open(fname, "r")
  examples = []
  for line in f:
    example = {}
    tokens = line.split("|")
    for t in tokens:
      if t.startswith("action "):
        example["action"] = int(t.split()[-1])
      if t.startswith("id "):
        example["id"] = int(t.split()[-1])
      if t.startswith("fulllabel "):
        example["label"] = int(t.split()[-1])
      if t.startswith("prop "):
        example["propensity"] = float(t.split()[-1])
      if t.startswith("loss "):
        example["loss"] = float(t.split()[-1])
      if t.startswith("probs "):
        example["probs"] = [float(p) for p in t.split()[1:]]

    examples.append(example)
    _test_consistency(examples)
    num_to_return = int(fraction * len(examples))
  return examples[:num_to_return]


def _test_consistency(examples):
  for ex in examples:
    assert ex["loss"] == (ex["action"] != ex["label"])
    assert ex["probs"][ex["action"]] == ex["propensity"]


def cifar_generator(blbf_mapfile, tmp_dir, training, fraction=1.0, pc=0.5):
  """Image generator for CIFAR-10 with BLBF.

  Args:
    blbf_mapfile: text file containing the logged data.
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    fraction: fraction of the examples in the log file to use.
    pc: Used to select the correct label probability in case no mapfile is None.

  Returns:
    An instance of image_generator that produces CIFAR-10 images and labels.
  """
  url = _CIFAR10_URL
  train_files = _CIFAR10_TRAIN_FILES
  test_files = _CIFAR10_TEST_FILES
  prefix = _CIFAR10_PREFIX
  image_size = _CIFAR10_IMAGE_SIZE
  label_key = "labels"

  _get_cifar(tmp_dir, url)
  data_files = train_files if training else test_files
  all_images = []
  all_labels = []
  all_actions = []
  all_rewards = []
  all_propensities = []

  for filename in data_files:
    path = os.path.join(tmp_dir, prefix, filename)
    with tf.gfile.Open(path, "rb") as f:
      if six.PY2:
        data = cPickle.load(f)
      else:
        data = cPickle.load(f, encoding="latin1")
    images = data["data"]
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, image_size, image_size))
    all_images.extend(
        [np.squeeze(images[j]).transpose((1, 2, 0)) for j in range(num_images)])
    labels = data[label_key]
    all_labels.extend([labels[j] for j in range(num_images)])

    actions = []
    rewards = []
    propensities = []

    for l in labels:
      if np.random.uniform() < pc:  # Correct label with pc probability.
        a = l  # rewrite logged action
      else:
        a = np.random.randint(10)  # Pick uniformly random action.
      actions.append(a)
      if a == l:
        rewards.append(1.0)
        propensities.append(pc + 0.1 * (1. - pc))
      else:
        rewards.append(0.0)
        propensities.append(pc * 0.1)

    all_actions.extend([actions[j] for j in range(num_images)])
    all_rewards.extend([rewards[j] for j in range(num_images)])
    all_propensities.extend([propensities[j] for j in range(num_images)])

  if blbf_mapfile is not None:
    blbf_images = []
    blbf_labels = []
    blbf_actions = []
    blbf_rewards = []
    blbf_propensities = []
    # In this case, load rewards/propensities/actions from the actual log data.
    # Read each line and append the example to all_*
    print("parsing the examples from the blbf mapfile")
    examples_blbf = _parse_blbf_file(blbf_mapfile, fraction)
    print("Parsed a total of {} examples from the blbf file".format(
        len(examples_blbf)))
    for ex in examples_blbf:
      blbf_images.append(all_images[ex["id"]])
      assert all_labels[ex["id"]] == ex["label"]
      blbf_labels.append(all_labels[ex["id"]])
      blbf_actions.append(ex["action"])
      blbf_rewards.append(1 - ex["loss"])
      blbf_propensities.append(ex["propensity"])
    all_images = blbf_images
    all_labels = blbf_labels
    all_actions = blbf_actions
    all_rewards = blbf_rewards
    all_propensities = blbf_propensities

  print("total samples in ds:")
  print(
      len(all_images), len(all_labels), len(all_actions), len(all_rewards),
      len(all_propensities))

  return blbf_utils.image_generator(all_images, all_labels, all_actions,
                                    all_rewards, all_propensities)


class ImageCifar10Tune(blbf_utils.Image2BanditProblem):
  """Cifar-10 Tune."""

  @property
  def is_small(self):
    return True

  @property
  def num_classes(self):
    return 10

  @property
  def train_shards(self):
    return 10

  ### copied from mnist-tune till above

  @property
  def num_channels(self):
    return 3

  @property
  def class_labels(self):
    return [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]

  def preprocess_example(self, example, mode, unused_hparams):
    image = example["inputs"]
    image.set_shape([_CIFAR10_IMAGE_SIZE, _CIFAR10_IMAGE_SIZE, 3])
    if mode == tf.estimator.ModeKeys.TRAIN:
      image = image_utils.cifar_image_augmentation(image)
    if not self._was_reversed:
      image = tf.image.per_image_standardization(image)
    example["inputs"] = image
    return example


@registry.register_problem
class ImageCifar10BlbfHalf(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = os.path.join(_CIFAR10_BLBF_PATH, _CIFAR_BLBF_MAP_FILES[0])
      print("reading half of the train data from from " + blbf_mapfile)
      return cifar_generator(blbf_mapfile, tmp_dir, True, fraction=0.5)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10Blbf1(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = os.path.join(_CIFAR10_BLBF_PATH, _CIFAR_BLBF_MAP_FILES[0])
      print("reading all of the train data from from " + blbf_mapfile)
      return cifar_generator(blbf_mapfile, tmp_dir, True)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10Blbf2(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = os.path.join(_CIFAR10_BLBF_PATH, _CIFAR_BLBF_MAP_FILES[1])
      return cifar_generator(blbf_mapfile, tmp_dir, True)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10Blbf3(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = os.path.join(_CIFAR10_BLBF_PATH, _CIFAR_BLBF_MAP_FILES[2])
      return cifar_generator(blbf_mapfile, tmp_dir, True)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10Blbf5(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = os.path.join(_CIFAR10_BLBF_PATH, _CIFAR_BLBF_MAP_FILES[3])
      return cifar_generator(blbf_mapfile, tmp_dir, True)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10BlbfLinearLogger0(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = os.path.join(_LINEAR_LOGGER_PATH, _LINEAR_MAP_FILES[0])
      return cifar_generator(blbf_mapfile, tmp_dir, True)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10BlbfGreedy10(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = None
      return cifar_generator(blbf_mapfile, tmp_dir, True, pc=0.1)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10BlbfGreedy30(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = None
      return cifar_generator(blbf_mapfile, tmp_dir, True, pc=0.3)
    else:
      return cifar_generator(None, tmp_dir, False)


@registry.register_problem
class ImageCifar10BlbfGreedy50(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      blbf_mapfile = None
      return cifar_generator(blbf_mapfile, tmp_dir, True, pc=0.5)
    else:
      return cifar_generator(None, tmp_dir, False)
