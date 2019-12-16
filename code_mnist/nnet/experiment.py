
"""Simple framework for running experiments:
     estimator = model + training_objective
     optimizer = step_sizer + optimization_update
       method = estimator + optimizer
         experiment = method + data + measurements
"""

import sys
from operator import itemgetter
import tensorflow as tf
import numpy as np
import nnet.measurement as meas


def model_eval_op(model, placeholders, input_inds, output_inds, target_inds,
                  evalfun):
  """Wrap a model with an evaluation function to form an evaluator"""
  model_outputs = (model(itemgetter(*input_inds)(placeholders))
                  if len(input_inds) == 1
                  else model(*itemgetter(*input_inds)(placeholders)))
  used_outputs = [model_outputs[o] for o in output_inds]
  placeholder_targets = [placeholders[t] for t in target_inds]
  return tf.reduce_sum(evalfun(*(used_outputs + placeholder_targets)))


class estimator():
  """wrap a model with a train loss to form an estimator"""
  def __init__(self, model, placeholders, input_inds, output_inds, target_inds,
               train_loss, regularize=0, critic=None):
    self.model = model
    self.placeholders = placeholders
    self.input_inds = input_inds
    self.output_inds = output_inds
    self.target_inds = target_inds
    # critic
    if critic is None:
      train_loss_op = model_eval_op(model, placeholders,
                                    input_inds, output_inds, target_inds,
                                    train_loss)
    else:
      critic_op, critic_output_inds, critic_target_inds = critic
      criticholders = list(placeholders) # replace placeholder with critic vals
      for (i, o) in enumerate(critic_output_inds):
        criticholders[critic_target_inds[i]] = critic_op[o]
      train_loss_op = model_eval_op(model, criticholders,
                                    input_inds, output_inds, target_inds,
                                    train_loss)
    # regularizer
    if regularize > 0:
      regularize_op = regularize*sum(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      self.train_obj = train_loss_op + regularize_op
    else:
      self.train_obj = train_loss_op


class optimizer():
  """Wrap a step_sizer with an optim_fun to form an optimizer"""
  def __init__(self, optim_fun, step_size_fun, name):
    with tf.variable_scope(name):
      self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32,
                                     name='global_step')
    self.step_sizer = step_size_fun(self.global_step)
    self.optim_fun = optim_fun(self.step_sizer)


class method():
  """Wrap an estimator with an optimizer to form a method"""
  def __init__(self, estimator, optimizer):
    self.estimator = estimator
    self.optimizer = optimizer
    self.train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=self.estimator.model.name)
    self.train_step = optimizer.optim_fun.minimize(
        estimator.train_obj,
        var_list=self.train_vars,
        global_step=optimizer.global_step)


def experiment(method, data_train, measures, gap_timer, sampler,
               outpath, echo=True, save_model=False,
               init_from_source=None, critic_init=None):
  """Run an experiment with _method_ on _data_ while capturing _measures_"""
    # setup
  saver = tf.train.Saver(method.train_vars)
  writer = tf.summary.FileWriter(outpath + '/summary')
  outfile = open(outpath + '/log.txt', 'w')
  results = np.zeros([gap_timer.maxupdate+1, len(measures)])
  num_components = min(len(data_train), len(method.estimator.placeholders))
  feed_dict = {}
    # session: init
  sess = tf.Session()
  meas.reset(measures)
  gap_timer.reset()
  u = 0
  if critic_init is not None:
    critic_restorer, critic_path = critic_init
    if critic_restorer is not None:
      critic_restorer.restore(sess, critic_path)
  if init_from_source is not None:
    source_restorer, source_path, copy_op = init_from_source
    if source_restorer is not None:
      source_restorer.restore(sess, source_path)
    if copy_op is not None:
      sess.run(copy_op)
  global_vars = tf.global_variables(scope=method.estimator.model.name)
  is_init = sess.run([tf.is_variable_initialized(v) for v in global_vars])
  uninitialized = [var for (var, init) in zip(global_vars, is_init) if not init]
  for var in uninitialized:
    sess.run(var.initializer)
  meas.update(measures)
  results[u, :] = meas.printout(method.estimator.model.name,
                                measures, sess, echo, outfile, writer, u)
    # session: loop
  while gap_timer.alive():
    inds = sampler.next_inds()
    for j in range(num_components):
      feed_dict[method.estimator.placeholders[j]] = data_train[j][inds]
    sess.run(method.train_step, feed_dict=feed_dict)
    if gap_timer.update():
      u += 1
      meas.update(measures)
      results[u, :] = meas.printout(method.estimator.model.name,
                                    measures, sess, echo, outfile, writer, u)
    # session: post
  if not echo:
    meas.printout(method.estimator.model.name, measures, sess, True)
  if save_model:
    savefile = saver.save(sess, outpath + '/saved_model/model')
    print('Saving to %s' % savefile)
  print
  outfile.close()
  return results, sess


def print_results(method, measures, results, outpath=None):
  outfile = sys.stdout if outpath is None else open(outpath + '/log.txt', 'a')
  updates, num_meas = results.shape
  for u in range(updates):
    outfile.write(method.estimator.model.name+'\t')
    for s in range(num_meas):
      outfile.write('%g\t'%(results[u, s]))
    outfile.write('\n')
  if outfile != sys.stdout:
    outfile.close()

