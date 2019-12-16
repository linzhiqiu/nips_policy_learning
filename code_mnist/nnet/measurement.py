
"""Simple framework for capturing various measurements recorded
   during an experiment"""


import time
import numpy as np
import tensorflow as tf


class measure():

  def __init__(self, label):
    self.label = label

  def update(self):
    None

  def reset(self):
    None

  def printout(self, sess, echo, outfile=None, writer=None, step=0):
    val = self.eval(sess, writer, step)
    if echo:
      valstr = '%s %g'%(self.label, val)
      if outfile is not None:
        outfile.write('%s %g '%(self.label, val))
    return valstr, val


class meas_eval(measure):

  def __init__(self, placeholders, data, eval_op, label):
    self.placeholders = placeholders
    self.eval_op = eval_op
    self.data = data
    self.num_components = min(len(data), len(placeholders))
    self.label = label
    self.summary = tf.summary.scalar(label, self.eval_op)

  def eval(self, sess, writer=None, step=0):
    feed_dict = {}
    for j in range(self.num_components):
      feed_dict[self.placeholders[j]] = self.data[j]
    [val, summary] = sess.run([self.eval_op, self.summary], feed_dict=feed_dict)
    if writer is not None:
      writer.add_summary(summary, step)
    return val


class meas_time(measure):

  def __init__(self, label):
    self.label = label
    self.start_time = None

  def update(self):
    if self.start_time is None:
      self.start_time = time.time()

  def reset(self):
    self.start_time = None

  def eval(self, sess, writer, step):
    return 0 if self.start_time is None else time.time() - self.start_time


class meas_iter(measure):

  def __init__(self, gap, label):
    self.gap = gap
    self.label = label
    self.iter = None

  def update(self):
    if self.iter is None:
      self.iter = 0
    else:
      self.iter += self.gap

  def reset(self):
    self.iter = None

  def eval(self, sess, writer, step):
    return self.iter


# general

def reset(meas_list):
  for meas in meas_list:
    meas.reset()

def update(meas_list):
  for meas in meas_list:
    meas.update()

def printout(label, meas_list, sess, echo, outfile=None, writer=None, step=0):
  results = np.zeros(len(meas_list))
  i = 0
  if echo:
    outstr = label
    if outfile is not None:
      outfile.write('%s '%label)
  for meas in meas_list:
    valstr, results[i] = meas.printout(sess, echo, outfile, writer, step)
    i += 1
    outstr += ' ' + valstr
  if echo:
    print(outstr)
    if outfile is not None:
      outfile.write('\n')
  return results

