
import nnet.util as util
import tensorflow as tf
import math


### utilities ###

def misclass_err(z_hat, y): # y is vector
  return tf.cast(tf.not_equal(tf.argmax(z_hat, 1), tf.argmax(y, 1)), tf.float32)

def sparse_misclass_err(z_hat, y): # y is index
  return tf.cast(tf.not_equal(tf.argmax(z_hat, 1), tf.cast(y, tf.int64)),
                 tf.float32)

def expected_value(z_hat, y, tau=1): # y is vector
  p_hat = tf.nn.softmax(z_hat/tau, axis=1)
  return tf.reduce_sum(tf.multiply(p_hat, y), axis=1)

def sparse_expected_value(z_hat, y, tau=1): # y is index, assume class indicator
  p_hat = tf.nn.softmax(z_hat/tau, axis=1)
  p_hat_y = util.on_indices(p_hat, y)
  return p_hat_y

def null(z_hat, y):
  return 0.0*z_hat[:, 0]


### full target information ###

# f0
def full_neg_expected_reward(z_hat, y, tau=1): # y is vector
  return 1 - expected_value(z_hat, y, tau)

def sparse_full_neg_expected_reward(z_hat, y, tau=1): # y is index
  return 1 - sparse_expected_value(z_hat, y, tau)

# f1
def full_squared_error(z_hat, y): # y is vector
  return tf.reduce_sum(tf.squared_difference(z_hat, y), axis=1)

def sparse_full_squared_error(z_hat, y): # y is index
  n = z_hat.get_shape().as_list()[1]
  y_mat = tf.one_hot(indices=y, depth=n)
  return full_squared_error(z_hat, y_mat)

# f2
def full_log_loss(z_hat, y): # y is vector
  return tf.nn.softmax_cross_entropy_with_logits(logits=z_hat, labels=y)

def sparse_full_log_loss(z_hat, y): # y is index
  return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z_hat, labels=y)

# f6
def full_kl_divergence_ml(z_hat, y, tau=1): # y is vector
  # decoupled form
  pot_z_hat = tf.reduce_logsumexp(z_hat, axis=1)
  if tau > 0:
    pot_y_tau = tf.reduce_logsumexp(y/tau, axis=1)
    p = tf.nn.softmax(y/tau, axis=1)
    return (pot_z_hat - pot_y_tau -
            tf.reduce_sum(tf.multiply(p, z_hat - y/tau), axis=1))
  else: # becomes equivalent to log_loss
    z_hat_y = tf.reduce_sum(tf.multiply(z_hat, y), axis=1)
    return pot_z_hat - z_hat_y

def sparse_full_kl_divergence_ml(z_hat, y, tau=1): # y is index
  # decoupled form
  pot_z_hat = tf.reduce_logsumexp(z_hat, axis=1)
  if tau > 0:
    n = z_hat.get_shape().as_list()[1]
    y_mat = tf.one_hot(indices=y, depth=n)
    pot_y_tau = tau*tf.reduce_logsumexp(y_mat/tau, axis=1)
    p = tf.nn.softmax(y_mat/tau, axis=1)
    return (pot_z_hat - pot_y_tau -
            tf.reduce_sum(tf.multiply(p, z_hat - y_mat/tau), axis=1))
  else: # becomes equivalent to log_loss
    z_hat_y = util.on_indices(z_hat, y)
    return pot_z_hat - z_hat_y

# f7
def full_combine(z_hat, y, alpha=1, beta=1, tau=1): # y is vector
  return ( (1 - alpha)*(1 - beta)*full_neg_expected_reward(z_hat, y, tau) +
           alpha*(1 - beta/2)*full_kl_divergence_ml(z_hat, y, tau) +
           (1 - alpha/2)*beta*full_squared_error(z_hat, y) )

def sparse_full_combine(z_hat, y, alpha=1, beta=1, tau=1): # y is index
  return ( (1 - alpha)*(1 - beta)*
             sparse_full_neg_expected_reward(z_hat, y, tau) +
           alpha*(1 - beta/2)*sparse_full_kl_divergence_ml(z_hat, y, tau) +
           (1 - alpha/2)*beta*sparse_full_squared_error(z_hat, y) )


### partial target information ###


  # expected reward: mode seeking

# e0
def sparse_part_neg_expected_reward(z_hat, a, r, p,
                                    shift=0, tau=1, xi=0, base=1):
  # a is index
  p_hat = tf.nn.softmax(z_hat/tau, axis=1)
  p_hat_a = util.on_indices(p_hat, a)
  return base + tf.multiply(tf.divide(p_hat_a, p) - xi, shift - r)


  # faking maximum likelihood: mode covering

# m0
def sparse_part_kl_ml_imputed(z_hat, a, r, p, tau=1): # a is index
  # decoupled form
  pot_z_hat = tf.reduce_logsumexp(z_hat, axis=1)
  z_hat_a = util.on_indices(z_hat, a)
  z_hat_max = tf.reduce_max(z_hat, axis=1)
  pot_r_impute = (tau*z_hat_max +
    tau*(tf.log(tf.reduce_sum(
                    tf.exp(z_hat - tf.expand_dims(z_hat_max, 1)), axis=1) -
                tf.exp(z_hat_a - z_hat_max) +
                tf.exp(r/tau - z_hat_max))))
  p_a = tf.exp((r - pot_r_impute)/tau)
  return pot_z_hat - pot_r_impute/tau + tf.multiply(p_a, r/tau - z_hat_a)


  # baseline regression losses: mode covering

# b0
def sparse_part_squared_error(z_hat, a, r, p): # a is index
  z_hat_a = util.on_indices(z_hat, a)
  return tf.squared_difference(z_hat_a, r)


  # composite losses

# c2
def sparse_part_combine(z_hat, a, r, p,
                        shift=0, alpha=1, beta=1, tau=1, base=1):
  return ( (1 - alpha)*(1 - beta)*
           sparse_part_neg_expected_reward(z_hat, a, r, p, shift, 1, base) +
           alpha*(1 - beta/2)*sparse_part_kl_ml_imputed(z_hat, a, r, p, tau) +
           (1 - alpha/2)*beta*sparse_part_squared_error(z_hat, a, r/tau, p) )

# c3
def sparse_part_combine_all(z_hat, a, r, p,
                            shift=0, alpha=1, beta=1, gamma=1, tau=1, base=1):
  if alpha != beta or alpha != gamma:
    total = alpha + beta + gamma
    alpha_p = alpha / total
    beta_p  = beta  / total
    gamma_p = gamma / total
  else:
    alpha_p = beta_p = gamma_p = 1.0/3.0
  return ( gamma_p * sparse_part_neg_expected_reward(z_hat, a, r, p, shift, 1,
                                                     base) +
           alpha_p * sparse_part_kl_ml_imputed(z_hat, a, r, p, tau) +
           beta_p * sparse_part_squared_error(z_hat, a, r/tau, p) )

