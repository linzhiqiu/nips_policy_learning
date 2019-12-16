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

"""Utility to define various loss combinations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf
FLAGS = flags.FLAGS
_NUM_CLASSES = 10


def surrogate_loss(logits, labels, tau=1.0):
  """Full information loss."""
  labels_1hot = tf.one_hot(labels, depth=_NUM_CLASSES)
  phat = tf.nn.softmax(labels_1hot * tau, axis=-1)
  t1 = tf.reduce_sum(phat * logits, axis=-1)
  t2 = tf.reduce_logsumexp(logits, axis=-1)
  return t2 - t1


def sq_loss(logits, labels, tau=1.0):
  """Full information loss."""
  labels_1hot = tf.one_hot(labels, depth=_NUM_CLASSES)
  targets = tf.cast(labels_1hot, tf.float32) * tau
  return tf.reduce_sum((targets - logits)**2, axis=-1)


def direct_loss(logits, labels, shift=0., tau_entropy=0.):
  """Full information expected reward loss."""
  pi = tf.nn.softmax(logits)
  labels_1hot = tf.one_hot(labels, depth=_NUM_CLASSES)
  reward = (labels_1hot - shift) * pi
  loss = -tf.reduce_sum(reward, axis=-1)
  if tau_entropy > 0:
    lse_q = tf.reduce_logsumexp(logits, axis=-1, keep_dims=True)
    log_prob_pi = logits - lse_q
    entropy_pi = tf.reduce_sum(-pi * log_prob_pi, axis=-1)
    loss += -tau_entropy * entropy_pi
  return loss


def coverage_loss(a, q):
  lse_q = tf.reduce_logsumexp(q, axis=-1, keep_dims=True)
  log_prob_pi = q - lse_q
  row_indices = tf.cast(tf.range(tf.shape(a)[0]), tf.int32)
  idx = tf.stack([row_indices, a], axis=1)
  logpia = tf.gather_nd(log_prob_pi, idx)
  return -logpia


def l2loss(a, r, p, q):
  """squared error loss used in partial feedback."""
  del p
  row_indices = tf.cast(tf.range(tf.shape(a)[0]), tf.int32)
  idx = tf.stack([row_indices, a], axis=1)
  qsa = tf.gather_nd(q, idx)
  return tf.reduce_sum(tf.squared_difference(qsa, r), axis=-1)


# Note that for tau=0, both pg losses are identical.
def pgloss1(a, r, p, q, tau=0.):
  """entropy regularized expected reward, with importance correction."""
  pi = tf.nn.softmax(q)
  lse_q = tf.reduce_logsumexp(q, axis=-1, keep_dims=True)
  log_prob_pi = q - lse_q
  entropy_pi = tf.reduce_sum(-pi * log_prob_pi, axis=-1)

  row_indices = tf.cast(tf.range(tf.shape(a)[0]), tf.int32)
  idx = tf.stack([row_indices, a], axis=1)
  pia = tf.gather_nd(pi, idx)
  rhat = tf.divide(pia, p) * r + tau * entropy_pi

  return -rhat


def pgloss2(a, r, p, q, tau=0.):
  """entropy regularized expected reward, with sampled entropy term."""
  pi = tf.nn.softmax(q)
  lse_q = tf.reduce_logsumexp(q, axis=-1, keep_dims=True)
  log_pi = q - lse_q
  row_indices = tf.cast(tf.range(tf.shape(a)[0]), tf.int32)
  idx = tf.stack([row_indices, a], axis=1)
  pia = tf.gather_nd(pi, idx)
  log_pia = tf.gather_nd(log_pi, idx)
  rhat = tf.divide(pia, p) * r - tau * pia * log_pia

  return -rhat


def soloss1(a, r, p, q):
  """Surrogate objective variant with full loss terms."""
  del p
  logging_phat = tf.ones_like(r)
  r_hat = doubly_robust_targets(q, a, r, logging_phat)
  t1 = tf.reduce_logsumexp(q, axis=-1)
  t3 = tf.reduce_logsumexp(r_hat, axis=-1)
  phat = tf.nn.softmax(r_hat)
  row_indices = tf.cast(tf.range(tf.shape(a)[0]), tf.int32)
  idx = tf.stack([row_indices, a], axis=1)
  phat_a = tf.gather_nd(phat, idx)
  zhat_a = tf.gather_nd(q, idx)
  t2 = phat_a * (r - zhat_a)
  loss = t1 + t2 - t3
  return loss


def soloss2(a, r, p, q):
  """Surrogate objective variant with stop gradient on imputed target."""
  del p
  logging_phat = tf.ones_like(r)
  r_hat = doubly_robust_targets(q, a, r, logging_phat)
  phat = tf.nn.softmax(r_hat)
  t1 = tf.reduce_logsumexp(q, axis=-1)
  t2 = tf.reduce_sum(phat * q, axis=-1)
  loss = t1 - t2
  return loss


def soloss3(a, r, p, q):
  """Surrogate objective variant with sampled loss term."""
  pi = tf.nn.softmax(q)
  a_one_hot = tf.one_hot(indices=a, depth=_NUM_CLASSES)
  # imputes the predictions everywhere except for the sampled action.
  r_imputed = a_one_hot * tf.expand_dims(r, 1) + (1 - a_one_hot) * q
  pi_r = tf.nn.softmax(r_imputed)

  row_indices = tf.cast(tf.range(tf.shape(a)[0]), tf.int32)
  idx = tf.stack([row_indices, a], axis=1)
  qsa = tf.gather_nd(q, idx)
  pia = tf.gather_nd(pi, idx)
  pi_ra = tf.gather_nd(pi_r, idx)
  loss = qsa * tf.stop_gradient(tf.divide(pia - pi_ra, p))

  return loss


def doubly_robust_targets(q_critic,
                          a_logged,
                          r_logged,
                          p_hat,
                          action_size=_NUM_CLASSES):
  """Forms the doubly robus target tensor from critic, action, reward, prop."""
  logged_avec = tf.one_hot(indices=a_logged, depth=action_size)
  # Copies the observed action to all dimensions and then it is masked by
  # logged_avec to compute the actual advantage.
  adv_brdcst = tf.divide(
      tf.expand_dims(r_logged, axis=-1) - q_critic,
      tf.expand_dims(p_hat, axis=-1))
  return logged_avec * adv_brdcst + q_critic
