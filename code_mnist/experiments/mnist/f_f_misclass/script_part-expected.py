
from absl import app
from absl import flags
import tensorflow as tf
import os

import task
import nnet.losses as losses

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # for macOS, and no this isn't okay

  # train loss
NAME = 'part-expected'
TRAIN_LOSS = (lambda z_hat, a, r, p:
                losses.sparse_part_neg_expected_reward(
                    z_hat, a, r, p, shift=FLAGS.shift, tau=FLAGS.tau,
                    xi=-1e-4, base=1))
EVAL_FUN = losses.sparse_misclass_err

  # source init
SOURCE_NAME = 'part-squared'
SOURCE_PATH = (SOURCE_NAME +
    '/step=0.02_reg=0.0_batch=200_valid=5000_sample=inf_0_epochs=10/0' +
    '/saved_model/model')

  # sample file
SAMPLE_TAU = 'inf'
SAMPLE_NUM = 0
SAMPLE_FILE = 'tau=' + SAMPLE_TAU + '/sample_' + str(SAMPLE_NUM) + '.txt'

# parameters
FLAGS = flags.FLAGS
flags.DEFINE_integer('label', 0, 'Experiment label')
flags.DEFINE_float('step', 0.02, 'Step size')
flags.DEFINE_float('reg', 0.0, 'Regularization')
flags.DEFINE_float('shift', 0.2, 'Offset')
flags.DEFINE_float('tau', 1.0, 'Temperature')
flags.DEFINE_integer('batch', 20, 'Mini-batch size')
flags.DEFINE_integer('valid', 5000, 'Validation set size')
flags.DEFINE_integer('epochs', 10, 'Training epochs')
flags.DEFINE_string('path', '.', 'Root for output path')
flags.DEFINE_boolean('save_model', False, 'Save model flag')
flags.DEFINE_boolean('echo', True, 'Echo flag')

def main(argv):
  paramstr = 'step=' + str(FLAGS.step) + '_reg=' + str(FLAGS.reg)
  paramstr += '_shift=' + str(FLAGS.shift)
  paramstr += '_tau=' + str(FLAGS.tau)
  paramstr += '_batch=' + str(FLAGS.batch) + '_valid=' + str(FLAGS.valid)
  paramstr += '_sample=' + SAMPLE_TAU + '_' + str(SAMPLE_NUM)
  paramstr += '_epochs=' + str(FLAGS.epochs)
  outpath = FLAGS.path + '/' + NAME + '/' + paramstr + '/' + str(FLAGS.label)
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  # run experiment
  task.experiment_part(NAME, TRAIN_LOSS, EVAL_FUN, FLAGS, outpath, SAMPLE_FILE,
                       init_source=(SOURCE_NAME, SOURCE_PATH))

if __name__ == '__main__':
  app.run(main)

