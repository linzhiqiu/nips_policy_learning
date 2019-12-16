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

"""Utilities for Batch Learning From Bandit Feedback (BLBF) image datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities
import tensorflow as tf


def image_generator(images, labels, actions, rewards, probabilities):
  """Generator for images that takes image and labels lists and creates pngs.

  Args:
    images: list of images given as [width x height x channels] numpy arrays.
    labels: list of ints, same length as images.
    actions: logged actions
    rewards: logged rewards
    probabilities: probabilities

  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the image as PNG,
    * image/format: the string "png" representing image format,
    * image/class/label: an integer representing the label,
    * image/height: an integer representing the height,
    * image/width: an integer representing the width.
    Every field is actually a singleton list of the corresponding type.

  Raises:
    ValueError: if images is an empty list.
  """
  if not images:
    raise ValueError("Must provide some images for the generator.")
  width, height, _ = images[0].shape
  for (enc_image, label, action, reward, probability) in zip(
      image_utils.encode_images_as_png(images), labels, actions, rewards,
      probabilities):
    yield {
        "image/encoded": [enc_image],
        "image/format": ["png"],
        "image/class/label": [int(label)],
        "image/class/action": [int(action)],
        "image/reward": [reward],
        "image/probability": [probability],
        "image/height": [height],
        "image/width": [width],
    }


class Image2BanditProblem(image_utils.ImageProblem):
  """Base class for image classification with partial feedback problems."""

  @property
  def is_small(self):
    raise NotImplementedError()

  @property
  def num_classes(self):
    raise NotImplementedError()

  @property
  def train_shards(self):
    raise NotImplementedError()

  @property
  def dev_shards(self):
    return 1

  @property
  def class_labels(self):
    return ["ID_%d" % i for i in range(self.num_classes)]

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.ImageEncoder(channels=self.num_channels),
        "targets": text_encoder.ClassLabelEncoder(self.class_labels)
    }

  def generator(self, data_dir, tmp_dir, is_training):
    raise NotImplementedError()

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (
        super(Image2BanditProblem, self).example_reading_spec())

    label_key = "image/class/label"
    data_fields[label_key] = tf.FixedLenFeature((1,), tf.int64)
    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(label_key)

    action_key = "image/class/action"
    data_fields[action_key] = tf.FixedLenFeature((1,), tf.int64)
    data_items_to_decoders[
        "actions"] = tf.contrib.slim.tfexample_decoder.Tensor(action_key)

    reward_key = "image/reward"
    data_fields[reward_key] = tf.FixedLenFeature((1,), tf.float32)
    data_items_to_decoders["reward"] = tf.contrib.slim.tfexample_decoder.Tensor(
        reward_key)

    probability_key = "image/probability"
    data_fields[probability_key] = tf.FixedLenFeature((1,), tf.float32)
    data_items_to_decoders[
        "probability"] = tf.contrib.slim.tfexample_decoder.Tensor(
            probability_key)

    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "inputs": modalities.ModalityType.IMAGE,
        "targets": modalities.ModalityType.CLASS_LABEL,
    }
    p.vocab_size = {"inputs": 256, "targets": self.num_classes}
    p.batch_size_multiplier = 4 if self.is_small else 256
    p.loss_multiplier = 3.0 if self.is_small else 1.0
    if self._was_reversed:
      p.loss_multiplier = 1.0
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE_LABEL

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))
