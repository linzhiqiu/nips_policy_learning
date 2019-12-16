"""Critics:
   given placeholders x, y, a, r, p,
   replace y with critic output
"""

from operator import itemgetter
import tensorflow as tf
from nnet import util
from nnet import losses


def passthrough(dims, placeholders):
  """Supervised null critic: but expands target action to indicator vector"""
  _, m = dims
  y = placeholders[1]
  y_mat = tf.nn.embedding_lookup(tf.eye(m), y)
  return y_mat, None


def impute_const(prior, target_inds, output_dim, placeholders):
  """Keeps observed rewards, imputes prior over unobserved actions"""
  m = output_dim
  a, r = itemgetter(*target_inds)(placeholders)
  if len(a.get_shape().as_list()) == 1: # scalar impute
    dims = tf.stack([tf.shape(a)[0], tf.constant(m)])
    prior = tf.cast(prior, tf.float32)
    mu_fill = tf.fill(dims, prior)
    inds = tf.reshape(tf.cumsum(tf.fill([dims[0]], m)) - m + a, (dims[0], 1))
    mu_diff = tf.reshape(tf.scatter_nd(inds, r - prior, [m*dims[0]]),
                         (dims[0], m))
    mu_posterior = mu_fill + mu_diff
  else: # quadratic impute
    diff = prior - a
    norms = tf.norm(diff, axis=1, keepdims=True)
    mu_posterior = a + tf.multiply(tf.expand_dims(tf.sqrt(-r), axis=1),
                                   tf.divide(diff, norms))
  return [mu_posterior], None


def model_predict(model, input_inds, output_inds, placeholders):
  """Critic model predictions"""
  model_outputs = (model(itemgetter(*input_inds)(placeholders))
                   if len(input_inds) == 1
                   else model(*itemgetter(*input_inds)(placeholders)))
  used_outputs = [tf.stop_gradient(model_outputs[o]) for o in output_inds]
  model_vars = tf.global_variables(scope=model.name)
  return used_outputs, model_vars


def posterior(model, input_inds, output_inds, target_inds, output_dim,
              prior_variances, placeholders):
  """Improve critic model predictions: posterior mean given observed rewards"""
  m = output_dim
  a, r = itemgetter(*target_inds)(placeholders)
  model_outputs = (model(itemgetter(*input_inds)(placeholders))
                  if len(input_inds) == 1
                  else model(*itemgetter(*input_inds)(placeholders)))
  used_outputs = [tf.stop_gradient(model_outputs[o]) for o in output_inds]
  if len(a.get_shape().as_list()) == 1: # multiclass posterior
    mu_prior = used_outputs[0]
    mu_prior_a = util.on_indices(mu_prior, a)
    dims = tf.stack([tf.shape(a)[0], tf.constant(m)])
    inds = tf.reshape(tf.cumsum(tf.fill([dims[0]], m)) - m + a, (dims[0], 1))
    mu_diff = tf.reshape(tf.scatter_nd(inds, r - mu_prior_a, [m*dims[0]]),
                         (dims[0], m))
    mu_posterior = mu_prior + mu_diff
    posterior = [mu_posterior]
  else: # quadratic posterior
    n = a.get_shape().as_list()[1]
    mu, sigma2, shift = used_outputs
    mvar2, mvar1, mvar0, noise = prior_variances
    mu_posterior, sigma2_posterior, shift_posterior = losses.gaussian_posterior(
        a, r, mu, sigma2, shift, mvar2, mvar1, mvar0, noise)
    shift_posterior = tf.zeros_like(r) # remove target shifts, policy invariant
    posterior = mu_posterior, sigma2_posterior, shift_posterior
  model_vars = tf.global_variables(scope=model.name)
  return posterior, model_vars

