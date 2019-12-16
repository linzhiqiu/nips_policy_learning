
# paths
import sys
HOMEPATH = '/Users/dale/'
SRCPATH = HOMEPATH + 'Documents/res/rl/neurips19/final/supplement/code_mnist'
DATAPATH = HOMEPATH + 'Documents/res/data/images/MNIST'
sys.path.append(SRCPATH)


# imports
import numpy as np
import tensorflow as tf
import nnet
from nnet import measurement as meas


# data and placeholders
def data_mnist(num_validation, sample_file=None):
  image_shape = [784]
    # read MNIST data: vector input format (used by fully connected model)
  data_train, data_valid, data_test = nnet.read_mnist.read_data(
      DATAPATH, image_shape=image_shape, image_range=[0.0, 1.0], one_hot=False,
      num_validation=num_validation)
    # dimensions
  t, n = data_train[0].shape
  te = data_test[1].shape[0]
  tv = num_validation
  m = 10
    # random action selection for training
  if sample_file is not None:
    sample_path = DATAPATH + '/sampled_labels/f_misclass/' + sample_file
    sample_train = nnet.read_mnist.read_apr_sample(sample_path)
    data_train_part = (data_train[0], data_train[1], sample_train[0][0:t],
                       sample_train[2][0:t], sample_train[1][0:t])
                       # x, y, a, r, p
  else:
    data_train_part = data_train
    # placeholders corresponding to data_train components
  placeholders = [tf.placeholder(tf.float32, shape=(None, image_shape[0])), # x
                  tf.placeholder(tf.int32,   shape=(None,)), # y
                  tf.placeholder(tf.int32,   shape=(None,)), # a
                  tf.placeholder(tf.float32, shape=(None,)), # r
                  tf.placeholder(tf.float32, shape=(None,))] # p
    # return
  data = data_train_part, data_valid, data_test
  dimensions = t, n, tv, te, m
  return data, dimensions, placeholders


# model
class model_f_f():
  """Define an f_f model: input -> fully connected -> output"""

  def __init__(self, name, dims):
    self.name = name
    self.dims = dims
    dim_in, dim_out = dims
    hidden = 512
    self.input_shape = (None, dim_in)
    self.output_shape = (None, dim_out)
    self.full1 = tf.layers.Dense(hidden,
                                 name=name,
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.nn.l2_loss)
    self.full2 = tf.layers.Dense(dim_out,
                                 name=name,
                                 kernel_regularizer=tf.nn.l2_loss)
    self.output_transfer = tf.nn.softmax

  def __call__(self, inputs):
    circuit = tf.layers.Flatten()(inputs)
    circuit = self.full1(circuit)
    z_hat = self.full2(circuit)
    y_hat = self.output_transfer(z_hat)
    return z_hat, y_hat


def source_init(source_name, dims):
  n, m = dims
  source_model = model_f_f(source_name, (n, m))
  source_outputs = source_model(tf.zeros((1, n)))
  source_vars = tf.trainable_variables(scope=source_name)
  source_restorer = tf.train.Saver(source_vars)
  return source_restorer

def copy_model(source_name, target_model):
  source_restorer = source_init(source_name, target_model.dims)
  copy_op = nnet.util.copy_model_vars(source_name, target_model)
  return source_restorer, copy_op


# method wrapper
def method_f_f_mom(name, dimensions, placeholders,
                   input_inds, output_inds, target_inds,
                   train_loss, regularization, stepsize, minibatch,
                   critic=None):
  """estimator = model + training_objective
     optimizer = step_sizer + optimization_update
     method = estimator + optimizer"""
  _, n, _, _, m = dimensions
    # estimator
  model = model_f_f(name, (n, m))
  estimator = nnet.experiment.estimator(
      model, placeholders, input_inds, output_inds, target_inds,
      train_loss, regularization, critic)
    # optimizer
  init_step = stepsize/minibatch
  step_size_fun = (lambda global_step:
                      nnet.step_sizers.const_step(init_step, global_step))
  optim_fun = lambda stepsize: tf.train.MomentumOptimizer(stepsize,
                                                          momentum=0.9)
  optimizer = nnet.experiment.optimizer(optim_fun, step_size_fun, name)
    # method
  return nnet.experiment.method(estimator, optimizer)


# experiment configs
def experiment_full(name, train_loss, eval_fun, FLAGS, outpath):
  """Experiment wrapper for training with fully observed data"""
  data, dimensions, placeholders = data_mnist(FLAGS.valid)
  data_train, data_valid, data_test = data
  epoch = (60000 - FLAGS.valid)//FLAGS.batch
  method = method_f_f_mom(name, dimensions, placeholders, [0], [0], [1],
                          train_loss, FLAGS.reg, FLAGS.step, FLAGS.batch)
  train_op = method.estimator.train_obj
  eval_op = nnet.experiment.model_eval_op(
      method.estimator.model, placeholders, [0], [0], [1], eval_fun)
  measures = (meas.meas_iter(epoch, 'step'),
              meas.meas_eval(placeholders, data_train, train_op, 'train_obj'),
              meas.meas_eval(placeholders, data_train, eval_op,  'train_err'),
              meas.meas_eval(placeholders, data_valid, eval_op,  'valid_err'),
              meas.meas_eval(placeholders, data_test,  eval_op,  'test_err'),
              meas.meas_time('train_time') )
  gap_timer = nnet.gap_timing.gaptimer(epoch, FLAGS.epochs)
  sampler = nnet.wrap_counting.wrapcounter(FLAGS.batch, dimensions[0])
  results, sess = nnet.experiment.experiment(
        method, data_train, measures, gap_timer, sampler, outpath,
        echo=FLAGS.echo, save_model=FLAGS.save_model)
  nnet.experiment.print_results(method, measures, results)
  nnet.experiment.print_results(method, measures, results, outpath)


def experiment_part(name, train_loss, eval_fun, FLAGS, outpath, sample_file,
                    init_source=None):
  """Experiment wrapper for training with partially observed data"""
  data, dimensions, placeholders = data_mnist(FLAGS.valid, sample_file)
  data_train, data_valid, data_test = data
  epoch = (60000 - FLAGS.valid)//FLAGS.batch
  method = method_f_f_mom(name, dimensions, placeholders, [0], [0], [2, 3, 4],
                          train_loss, FLAGS.reg, FLAGS.step, FLAGS.batch)
  train_op = method.estimator.train_obj
  eval_op = nnet.experiment.model_eval_op(
      method.estimator.model, placeholders, [0], [0], [1], eval_fun)
  init_from_source = None
  if init_source is not None:
    source_name, source_path = init_source
    source_restorer, copy_op = copy_model(source_name, method.estimator.model)
    init_from_source = source_restorer, source_path, copy_op
  measures = (meas.meas_iter(epoch, 'step'),
              meas.meas_eval(placeholders, data_train, train_op, 'train_obj'),
              meas.meas_eval(placeholders, data_train, eval_op,  'train_err'),
              meas.meas_eval(placeholders, data_valid, eval_op,  'valid_err'),
              meas.meas_eval(placeholders, data_test,  eval_op,  'test_err'),
              meas.meas_time('train_time') )
  gap_timer = nnet.gap_timing.gaptimer(epoch, FLAGS.epochs)
  sampler = nnet.wrap_counting.wrapcounter(FLAGS.batch, dimensions[0])
  results, sess = nnet.experiment.experiment(
        method, data_train, measures, gap_timer, sampler, outpath,
        echo=FLAGS.echo, save_model=FLAGS.save_model,
        init_from_source=init_from_source)
  nnet.experiment.print_results(method, measures, results)
  nnet.experiment.print_results(method, measures, results, outpath)


def experiment_actr(name, train_loss, eval_fun, FLAGS, outpath, sample_file,
                    critic=None, init_source=None):
  """Experiment wrapper for policy optimization using critic predictions"""
  data, dimensions, placeholders = data_mnist(FLAGS.valid, sample_file)
  data_train, data_valid, data_test = data
  t, n, _, _, m = dimensions
  epoch = (60000 - FLAGS.valid)//FLAGS.batch

    # critic selection
  critic_type, critic_model, critic_par, init_from_critic = critic
  if critic_type == 'impute':
    critic_path = None
    impute_value = critic_par
    critic_op, critic_vars = nnet.critics.impute_const(
        impute_value, [2, 3], m, placeholders)
    critic_output_inds = [0]
    critic_target_inds = [1]
    critic_restorer = None
  elif critic_type == 'model':
    critic_path = critic_par
    model = model_f_f(critic_model, (n, m))
    critic_op, critic_vars = nnet.critics.model_predict(
        model, [0], [0], placeholders)
    critic_output_inds = [0]
    critic_target_inds = [1]
    critic_restorer = tf.train.Saver(critic_vars)
  elif critic_type == 'posterior':
    critic_path = critic_par
    model = model_f_f(critic_model, (n, m))
    critic_op, critic_vars = nnet.critics.posterior(
        model, [0], [0], [2, 3], m, None, placeholders)
    critic_output_inds = [0]
    critic_target_inds = [1]
    critic_restorer = tf.train.Saver(critic_vars)
  critic = critic_op, critic_output_inds, critic_target_inds
  critic_init = critic_restorer, critic_path

  method = method_f_f_mom(
      name, dimensions, placeholders, [0], [0], [1], train_loss,
      FLAGS.reg, FLAGS.step, FLAGS.batch, critic)
  train_op = method.estimator.train_obj
  eval_op = nnet.experiment.model_eval_op(
      method.estimator.model, placeholders, [0], [0], [1], eval_fun)
  if init_from_critic:
    copy_op = nnet.util.copy_model_vars(critic_model, method.estimator.model)
    init_from_source = None, None, copy_op
  else:
    init_from_source = None
    if init_source is not None:
      source_name, source_path = init_source
      source_restorer, copy_op = copy_model(source_name, method.estimator.model)
      init_from_source = source_restorer, source_path, copy_op
  measures = (meas.meas_iter(epoch, 'step'),
              meas.meas_eval(placeholders, data_train, train_op, 'train_obj'),
              meas.meas_eval(placeholders, data_train, eval_op,  'train_err'),
              meas.meas_eval(placeholders, data_valid, eval_op,  'valid_err'),
              meas.meas_eval(placeholders, data_test,  eval_op,  'test_err'),
              meas.meas_time('train_time') )
  gap_timer = nnet.gap_timing.gaptimer(epoch, FLAGS.epochs)
  sampler = nnet.wrap_counting.wrapcounter(FLAGS.batch, t)
  results, sess = nnet.experiment.experiment(
        method, data_train, measures, gap_timer, sampler, outpath,
        echo=FLAGS.echo, save_model=FLAGS.save_model,
        init_from_source=init_from_source, critic_init=critic_init)
  nnet.experiment.print_results(method, measures, results)
  nnet.experiment.print_results(method, measures, results, outpath)
