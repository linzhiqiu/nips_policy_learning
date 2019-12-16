
from absl import app
from absl import flags
import tensorflow as tf
import os

import task
import nnet.losses as losses

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # for macOS, and no this isn't okay

  # train loss
NAME = 'full-combine'
TRAIN_LOSS = (lambda z_hat, y:
              losses.sparse_full_combine(z_hat, y,
                alpha=FLAGS.alpha, beta=FLAGS.beta, tau=FLAGS.tau))
EVAL_FUN = losses.sparse_misclass_err

  # experimental configuration
FLAGS = flags.FLAGS
flags.DEFINE_integer('label', 0, 'Experiment label')
flags.DEFINE_float('step', 0.1, 'Step size')
flags.DEFINE_float('reg', 0.0, 'Regularization')
flags.DEFINE_float('alpha', 1.0, 'Mix')
flags.DEFINE_float('beta', 0.0, 'Mix')
flags.DEFINE_float('tau', 0.2, 'Temperature')
flags.DEFINE_integer('batch', 100, 'Mini-batch size')
flags.DEFINE_integer('valid', 5000, 'Validation set size')
flags.DEFINE_integer('epochs', 10, 'Training epochs')
flags.DEFINE_string('path', '.', 'Root for output path')
flags.DEFINE_boolean('save_model', False, 'Save model flag')
flags.DEFINE_boolean('echo', True, 'Echo flag')

def main(argv):
  paramstr = 'step=' + str(FLAGS.step) + '_reg=' + str(FLAGS.reg)
  paramstr += '_alpha=' + str(FLAGS.alpha) + '_beta=' + str(FLAGS.beta)
  paramstr += '_tau=' + str(FLAGS.tau)
  paramstr += '_batch=' + str(FLAGS.batch) + '_valid=' + str(FLAGS.valid)
  paramstr += '_epochs=' + str(FLAGS.epochs)
  outpath = FLAGS.path + '/' + NAME + '/' + paramstr + '/' + str(FLAGS.label)
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  # run experiment
  task.experiment_full(NAME, TRAIN_LOSS, EVAL_FUN, FLAGS, outpath)

if __name__ == '__main__':
  app.run(main)

