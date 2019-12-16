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

"""Resnets with batch learning from bandit feedback."""
# Copied from cloud_tpu/models/resnet/resnet_model.py and modified

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow as tf
from off_policy_optimization.cifar.t2t import losses

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def layers():
  return common_layers.layers()


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    data_format="channels_first"):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == "channels_first":
    axis = 1
  else:
    axis = 3

  inputs = layers().BatchNormalization(
      axis=axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      fused=True,
      gamma_initializer=gamma_initializer)(
          inputs, training=is_training)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format="channels_first"):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
      height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == "channels_first":
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format="channels_first",
                         use_td=False,
                         targeting_rate=None,
                         keep_prob=None,
                         is_training=None):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.
    is_training: `bool` for whether the model is in training.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.

  Raises:
    Exception: if use_td is not valid.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  if use_td:
    inputs_shape = common_layers.shape_list(inputs)
    if use_td == "weight":
      if data_format == "channels_last":
        size = kernel_size * kernel_size * inputs_shape[-1]
      else:
        size = kernel_size * kernel_size * inputs_shape[1]
      targeting_count = targeting_rate * tf.to_float(size)
      targeting_fn = common_layers.weight_targeting
    elif use_td == "unit":
      targeting_count = targeting_rate * filters
      targeting_fn = common_layers.unit_targeting
    else:
      raise Exception("Unrecognized targeted dropout type: %s" % use_td)

    y = common_layers.td_conv(
        inputs,
        filters,
        kernel_size,
        targeting_count,
        targeting_fn,
        keep_prob,
        is_training,
        do_prune=True,
        strides=strides,
        padding=("SAME" if strides == 1 else "VALID"),
        data_format=data_format,
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer())
  else:
    y = layers().Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=("SAME" if strides == 1 else "VALID"),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)(
            inputs)

  return y


def residual_block(inputs,
                   filters,
                   is_training,
                   projection_shortcut,
                   strides,
                   final_block,
                   data_format="channels_first",
                   use_td=False,
                   targeting_rate=None,
                   keep_prob=None):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
      the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    projection_shortcut: `function` to use for projection shortcuts (typically a
      1x1 convolution to match the filter dimensions). If None, no projection is
      used and the input is passed as unchanged through the shortcut connection.
    strides: `int` block stride. If greater than 1, this block will ultimately
      downsample the input.
    final_block: unused parameter to keep the same function signature as
      `bottleneck_block`.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    The output `Tensor` of the block.
  """
  del final_block
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  return inputs + shortcut


def bottleneck_block(inputs,
                     filters,
                     is_training,
                     projection_shortcut,
                     strides,
                     final_block,
                     data_format="channels_first",
                     use_td=False,
                     targeting_rate=None,
                     keep_prob=None):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
      the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    projection_shortcut: `function` to use for projection shortcuts (typically a
      1x1 convolution to match the filter dimensions). If None, no projection is
      used and the input is passed as unchanged through the shortcut connection.
    strides: `int` block stride. If greater than 1, this block will ultimately
      downsample the input.
    final_block: `bool` set to True if it is this the final block in the group.
      This is changes the behavior of batch normalization initialization for the
      final batch norm in a block.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)

  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob,
      is_training=is_training)
  inputs = batch_norm_relu(
      inputs,
      is_training,
      relu=False,
      init_zero=final_block,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_layer(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training,
                name,
                data_format="channels_first",
                use_td=False,
                targeting_rate=None,
                keep_prob=None):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
      greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    """Project identity branch."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format,
        use_td=use_td,
        targeting_rate=targeting_rate,
        keep_prob=keep_prob,
        is_training=is_training)
    return batch_norm_relu(
        inputs, is_training, relu=False, data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(
      inputs,
      filters,
      is_training,
      projection_shortcut,
      strides,
      False,
      data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob)

  for i in range(1, blocks):
    inputs = block_fn(
        inputs,
        filters,
        is_training,
        None,
        1, (i + 1 == blocks),
        data_format,
        use_td=use_td,
        targeting_rate=targeting_rate,
        keep_prob=keep_prob)

  return tf.identity(inputs, name)


def resnet_v2(inputs,
              block_fn,
              layer_blocks,
              filters,
              data_format="channels_first",
              is_training=False,
              is_cifar=True,
              use_td=False,
              targeting_rate=None,
              keep_prob=None):
  """Resnet model.

  Args:
    inputs: `Tensor` images.
    block_fn: `function` for the block to use within the model. Either
      `residual_block` or `bottleneck_block`.
    layer_blocks: list of 3 or 4 `int`s denoting the number of blocks to include
      in each of the 3 or 4 block groups. Each group consists of blocks that
      take inputs of the same resolution.
    filters: list of 4 or 5 `int`s denoting the number of filter to include in
      block.
    data_format: `str`, "channels_first" `[batch, channels, height, width]` or
      "channels_last" `[batch, height, width, channels]`.
    is_training: bool, build in training mode or not.
    is_cifar: bool, whether the data is CIFAR or not.
    use_td: `str` one of "weight" or "unit". Set to False or "" to disable
      targeted dropout.
    targeting_rate: `float` proportion of weights to target with targeted
      dropout.
    keep_prob: `float` keep probability for targeted dropout.

  Returns:
    Pre-logit activations.
  """
  inputs = block_layer(
      inputs=inputs,
      filters=filters[1],
      block_fn=block_fn,
      blocks=layer_blocks[0],
      strides=1,
      is_training=is_training,
      name="block_layer1",
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob)
  inputs = block_layer(
      inputs=inputs,
      filters=filters[2],
      block_fn=block_fn,
      blocks=layer_blocks[1],
      strides=2,
      is_training=is_training,
      name="block_layer2",
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob)
  inputs = block_layer(
      inputs=inputs,
      filters=filters[3],
      block_fn=block_fn,
      blocks=layer_blocks[2],
      strides=2,
      is_training=is_training,
      name="block_layer3",
      data_format=data_format,
      use_td=use_td,
      targeting_rate=targeting_rate,
      keep_prob=keep_prob)
  if not is_cifar:
    inputs = block_layer(
        inputs=inputs,
        filters=filters[4],
        block_fn=block_fn,
        blocks=layer_blocks[3],
        strides=2,
        is_training=is_training,
        name="block_layer4",
        data_format=data_format,
        use_td=use_td,
        targeting_rate=targeting_rate,
        keep_prob=keep_prob)

  return inputs


@registry.register_model
class ResnetPartialFeedback(t2t_model.T2TModel):
  """Residual Network with partial feedback training."""

  def body(self, features):
    hp = self.hparams
    block_fns = {
        "residual": residual_block,
        "bottleneck": bottleneck_block,
    }
    assert hp.block_fn in block_fns
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    if is_training:
      targets = features["targets_raw"]

    inputs = features["inputs"]

    data_format = "channels_last"
    if hp.use_nchw:
      # Convert from channels_last (NHWC) to channels_first (NCHW). This
      # provides a large performance boost on GPU.
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
      data_format = "channels_first"

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=hp.filter_sizes[0],
        kernel_size=7,
        strides=1 if hp.is_cifar else 2,
        data_format=data_format)
    inputs = tf.identity(inputs, "initial_conv")
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

    if not hp.is_cifar:
      inputs = layers().MaxPooling2D(
          pool_size=3, strides=2, padding="SAME", data_format=data_format)(
              inputs)

      inputs = tf.identity(inputs, "initial_max_pool")

    out = resnet_v2(
        inputs,
        block_fns[hp.block_fn],
        hp.layer_sizes,
        hp.filter_sizes,
        data_format,
        is_training=is_training,
        is_cifar=hp.is_cifar,
        use_td=hp.use_td,
        targeting_rate=hp.targeting_rate,
        keep_prob=hp.keep_prob)

    if hp.use_nchw:
      out = tf.transpose(out, [0, 2, 3, 1])

    if not hp.is_cifar:
      return out

    out = tf.reduce_mean(out, [1, 2])
    num_classes = self._problem_hparams.vocab_size["targets"]
    if hasattr(self._hparams, "vocab_divisor"):
      num_classes += (-num_classes) % self._hparams.vocab_divisor
    logits = layers().Dense(num_classes, name="logits")(out)

    if hp.split_actor:
      actor_logits = layers().Dense(
          num_classes, name="actor_logits")(
              tf.stop_gradient(out))

    loss_dict = {"training": 0.0}
    if is_training:
      if hp.full_baseline or hp.surrogate_baseline or hp.pg_baseline or hp.l2_baseline or hp.mixture_baseline:
        labels_ = tf.reshape(targets, [-1])
        if hp.full_baseline:
          loss = tf.losses.sparse_softmax_cross_entropy(
              labels=labels_, logits=logits)
        elif hp.surrogate_baseline:
          # We want to do train with surrogate rather than xent.
          loss = losses.surrogate_loss(logits, labels_, tau=hp.tau_surrogate)
        elif hp.pg_baseline:
          loss = losses.direct_loss(
              logits,
              labels_,
              shift=hp.reward_baseline,
              tau_entropy=hp.tau_entropy)
        elif hp.l2_baseline:
          loss = losses.sq_loss(logits, labels_)
        elif hp.mixture_baseline:
          # Hardcoding the temperature of 0.1
          t1 = hp.scale_l2 * losses.sq_loss(logits, labels_, hp.tau_surrogate)
          t2 = losses.direct_loss(
              logits,
              labels_,
              shift=hp.reward_baseline,
              tau_entropy=hp.tau_entropy)
          t3 = losses.surrogate_loss(logits, labels_, hp.tau_surrogate)
          loss = hp.full_alpha * t1 + hp.full_beta * t2 + hp.full_gamma * t3
        loss_tensor = tf.reduce_mean(loss)
        loss_dict = {"training": loss_tensor}
      else:
        # Uncomment this to stay with the extra dims when needed.
        q = tf.squeeze(logits)
        a = tf.squeeze(features["actions"])
        r = hp.reward_scale * (
            tf.squeeze(features["reward"]) - hp.reward_baseline)
        p = tf.squeeze(features["probability"])
        if hp.loss_type == "l2":
          loss_tensor = losses.l2loss(a, r, p, q)
        if hp.loss_type == "pg":
          loss_tensor = losses.pgloss1(a, r, p, q, tau=hp.tau_entropy)
        if hp.loss_type == "so":
          loss_tensor = losses.soloss3(a, r, p, q)
        if hp.loss_type == "so1":
          loss_tensor = losses.soloss1(a, r, p, q)
        elif hp.loss_type == "combo":
          loss_tensor = hp.alpha * hp.scale_l2 * losses.l2loss(a, r, p, q) + \
              hp.beta * losses.pgloss1(a, r, p, q) + \
              hp.gamma * losses.soloss3(a, r, p, q)

        loss_tensor += hp.coverage_scale * losses.coverage_loss(a, q)

        if hp.split_actor:
          q_actor = tf.squeeze(actor_logits)
          q_critic = q
          if hp.actor_loss == "pg":
            loss_tensor_actor = losses.actor_pgloss(q_actor, q_critic, a, r, p)
          elif hp.actor_loss == "so":
            loss_tensor_actor = losses.actor_soloss(q_actor, q_critic, a, r, p)
          loss_tensor += loss_tensor_actor

        loss_dict = {"training": tf.reduce_mean(loss_tensor)}

    if hp.split_actor:
      logits = actor_logits  # rewrite the metrics as per actor head.

    logits = tf.reshape(logits, [-1, 1, 1, 1, logits.shape[1]])

    return logits, loss_dict

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Predict."""
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None
    logits, _ = self(features)  # pylint: disable=not-callable
    assert len(logits.get_shape()) == 5
    logits = tf.squeeze(logits, [1, 2, 3])
    log_probs = common_layers.log_prob_from_logits(logits)
    predictions, scores = common_layers.argmax_with_score(log_probs)
    return {"outputs": predictions, "scores": scores, "log_probs": log_probs}


def resnet_base():
  """Set of hyperparameters."""
  # For imagenet on TPU:
  # Set train_steps=120000
  # Set eval_steps=48

  # Base
  hparams = common_hparams.basic_params1()

  # Model-specific parameters
  hparams.add_hparam("layer_sizes", [3, 4, 6, 3])
  hparams.add_hparam("filter_sizes", [64, 64, 128, 256, 512])
  hparams.add_hparam("block_fn", "bottleneck")
  hparams.add_hparam("use_nchw", False)
  hparams.add_hparam("is_cifar", True)

  # Targeted dropout
  hparams.add_hparam("use_td", False)
  hparams.add_hparam("targeting_rate", None)
  hparams.add_hparam("keep_prob", None)

  # Variable init
  hparams.initializer = "normal_unit_scaling"
  hparams.initializer_gain = 2.

  # Optimization
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  hparams.optimizer_momentum_nesterov = True
  hparams.weight_decay = 1e-4
  hparams.clip_grad_norm = 0.0
  # (base_lr=0.1) * (batch_size=128*8 (on TPU, or 8 GPUs)=1024) / (256.)
  hparams.learning_rate = 0.1
  hparams.learning_rate_decay_scheme = "cosine"
  # For image_imagenet224, 120k training steps, which effectively makes this a
  # cosine decay (i.e. no cycles).
  hparams.learning_rate_cosine_cycle_steps = 120000
  hparams.batch_size = 128
  return hparams


def resnet_cifar_partial_32():
  """Set of hyperparameters."""
  hp = resnet_base()
  hp.add_hparam("split_actor", False)  # Use a split actor/critic model
  hp.add_hparam("actor_loss", "pg")
  hp.block_fn = "residual"
  hp.is_cifar = True
  hp.layer_sizes = [3, 3, 3]  # 20
  hp.filter_sizes = [16, 32, 64, 128]
  return hp


def resnet_cifar_partial():
  """Baseline parameters."""
  hp = resnet_cifar_partial_32()
  hp.add_hparam("full_baseline", False)
  hp.add_hparam("l2_baseline", False)
  hp.add_hparam("reward_baseline", 0.0)
  hp.add_hparam("reward_scale", 1.0)
  hp.add_hparam("pg_baseline", False)
  hp.add_hparam("surrogate_baseline", False)
  hp.add_hparam("tau_surrogate", 1.0)
  hp.add_hparam("mixture_baseline", False)
  hp.add_hparam("coverage_scale", 0.0)
  hp.add_hparam("full_alpha", 0.25)
  hp.add_hparam("full_beta", 0.)
  hp.add_hparam("full_gamma", 0.75)
  hp.add_hparam("tau_entropy", 0.0)
  hp.add_hparam("scale_l2", 0.01)
  return hp


@registry.register_hparams
def resnet_baseline():
  hp = resnet_cifar_partial()
  hp.full_baseline = True
  return hp


@registry.register_hparams
def resnet_baseline_l2():
  hp = resnet_cifar_partial()
  hp.full_baseline = False
  hp.l2_baseline = True
  hp.learning_rate = 0.01
  return hp


@registry.register_hparams
def resnet_baseline_pg():
  hp = resnet_cifar_partial()
  hp.full_baseline = False
  hp.pg_baseline = True
  hp.reward_baseline = 0.00
  return hp


@registry.register_hparams
def resnet_baseline_kl_rl():
  hp = resnet_cifar_partial()
  hp.full_baseline = False
  hp.pg_baseline = True
  hp.reward_baseline = 0.2
  hp.tau_entropy = 0.01
  return hp


@registry.register_hparams
def resnet_baseline_surrogate1():
  hp = resnet_cifar_partial()
  hp.surrogate_baseline = True
  hp.tau_surrogate = 1.0
  return hp


@registry.register_hparams
def resnet_baseline_surrogate10():
  hp = resnet_cifar_partial()
  hp.surrogate_baseline = True
  hp.tau_surrogate = 10.0
  return hp


@registry.register_hparams
def resnet_baseline_surrogate05():
  hp = resnet_cifar_partial()
  hp.surrogate_baseline = True
  hp.tau_surrogate = 5.0
  return hp


@registry.register_hparams
def resnet_baseline_surrogate01():
  hp = resnet_cifar_partial()
  hp.surrogate_baseline = True
  hp.tau_surrogate = 0.1
  return hp


@registry.register_hparams
def resnet_baseline_mixture():
  hp = resnet_cifar_partial()
  hp.mixture_baseline = True
  hp.tau_surrogate = 10.0
  return hp


@registry.register_hparams
def resnet_baseline_mixture2():
  hp = resnet_cifar_partial()
  hp.mixture_baseline = True
  hp.tau_surrogate = 10.0
  hp.full_alpha = 0.01
  hp.full_beta = 0.001
  hp.full_gamma = 0.99
  return hp


@registry.register_hparams
def resnet_baseline_mixture3():
  hp = resnet_cifar_partial()
  hp.mixture_baseline = True
  hp.tau_surrogate = 10.0
  hp.full_alpha = 0.025
  hp.full_beta = 0.95
  hp.full_gamma = 0.025
  return hp


@registry.register_hparams
def resnet_baseline_mixture4():
  hp = resnet_cifar_partial()
  hp.mixture_baseline = True
  hp.tau_surrogate = 10.0
  hp.full_alpha = 0.33
  hp.full_beta = 0.33
  hp.full_gamma = 0.33
  return hp


@registry.register_hparams
def resnet_baseline_mixture5():
  hp = resnet_cifar_partial()
  hp.mixture_baseline = True
  hp.tau_surrogate = 10.0
  hp.full_alpha = 0.
  hp.full_beta = 0.5
  hp.full_gamma = 0.5
  return hp


@registry.register_hparams
def resnet_split_actor_pg():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "l2")  # critic loss implicit l2 regression
  hparams.split_actor = True  # use the split model
  hparams.actor_loss = "pg"
  return hparams


@registry.register_hparams
def resnet_split_actor_so():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "l2")  # critic loss implicit l2 regression
  hparams.split_actor = True  # use the split model
  hparams.actor_loss = "so"
  return hparams


@registry.register_hparams
def resnet_l2():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "l2")
  hparams.learning_rate = 0.001
  return hparams


@registry.register_hparams
def resnet_pg():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  return hparams


@registry.register_hparams
def resnet_so():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "so")
  hparams.reward_scale = 10.0
  return hparams


@registry.register_hparams
def resnet_so1():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "so1")
  hparams.reward_scale = 10.0
  return hparams


@registry.register_hparams
def resnet_combo():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "combo")
  hparams.add_hparam("alpha", 0.33)
  hparams.add_hparam("beta", 0.33)
  hparams.add_hparam("gamma", 0.33)
  hparams.reward_scale = 10.0
  return hparams


@registry.register_hparams
def resnet_combo_02_95_02():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "combo")
  hparams.add_hparam("alpha", 0.025)
  hparams.add_hparam("beta", 0.95)
  hparams.add_hparam("gamma", 0.025)
  hparams.reward_scale = 10.0
  return hparams


@registry.register_hparams
def resnet_combo12():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "combo")
  hparams.add_hparam("alpha", 0.1)
  hparams.add_hparam("beta", 0.9)
  hparams.add_hparam("gamma", 0.)
  return hparams


@registry.register_hparams
def resnet_combo23():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "combo")
  hparams.add_hparam("alpha", 0.)
  hparams.add_hparam("beta", 0.9)
  hparams.add_hparam("gamma", 0.1)
  hparams.reward_scale = 10.0
  return hparams


@registry.register_hparams
def resnet_combo23_coverage():
  hparams = resnet_combo23()
  hparams.coverage_scale = 0.1
  return hparams


@registry.register_hparams
def resnet_combo13():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "combo")
  hparams.add_hparam("alpha", 0.25)
  hparams.add_hparam("beta", 0.)
  hparams.add_hparam("gamma", 1.0)
  hparams.reward_scale = 10.0
  return hparams


@registry.register_hparams
def resnet_pg_b05():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.05
  return hparams


@registry.register_hparams
def resnet_pg_b1():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.1
  return hparams


@registry.register_hparams
def resnet_kl_rl_b05():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.05
  hparams.tau_entropy = 0.01
  return hparams


@registry.register_hparams
def resnet_kl_rl_b1():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.1
  hparams.tau_entropy = 0.1
  return hparams


@registry.register_hparams
def resnet_kl_rl():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.tau_entropy = 0.001
  return hparams


@registry.register_hparams
def resnet_pg_b15():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.15
  return hparams


@registry.register_hparams
def resnet_pg_b2():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.2
  return hparams


@registry.register_hparams
def resnet_pg_b4():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.4
  return hparams


@registry.register_hparams
def resnet_pg_b6():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.6
  return hparams


@registry.register_hparams
def resnet_pg_b8():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 0.8
  return hparams


@registry.register_hparams
def resnet_pg_b10():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "pg")
  hparams.reward_baseline = 1.0
  return hparams


@registry.register_hparams
def resnet_so_rt1():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "so")
  hparams.reward_scale = 1.0
  return hparams


@registry.register_hparams
def resnet_so_rt2():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "so")
  hparams.reward_scale = 5.0
  return hparams


@registry.register_hparams
def resnet_so_rt3():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "so")
  hparams.reward_scale = 10.0
  return hparams


@registry.register_hparams
def resnet_so_rt4():
  hparams = resnet_cifar_partial()
  hparams.add_hparam("loss_type", "so")
  hparams.reward_scale = 15.0
  return hparams
