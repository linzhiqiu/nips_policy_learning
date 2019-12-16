
import numpy as np

"""Simple wrap counter: grabs chunks of indices, repermuted after every pass"""

class wrapcounter():

  def __init__(self, gap, length, shuffle=True, seed=None):
    self.gap = gap
    self.length = length
    self.order = np.arange(length)
    if shuffle:
      np.random.seed(seed=seed)
      np.random.shuffle(self.order)
    self.start = 0
    self.wraps = 0

  def next_inds(self, seed=None):
    start = self.start
    end = start + self.gap
    if end >  self.length:
      self.wraps += 1
      self.start = start = 0
      end = start + self.gap
      np.random.shuffle(self.order)
    self.start += self.gap
    return self.order[start:end]


"""Simple run counter: grabs chunks of indices in order, partitioned by batch"""

class runcounter():

  def __init__(self, batch, steps, length):
    self.batch = batch
    self.steps = steps
    self.num_batches = length // batch // steps
    self.sub_length = steps * self.num_batches
    self.length = batch * self.sub_length
    self.sequence = np.arange(self.length)
    self.subsequences = np.reshape(self.sequence, [self.batch, self.sub_length])
    self.start = 0
    self.wraps = 0

  def next_inds(self, seed=None):
    start = self.start
    end = start + self.steps
    if end > self.sub_length:
      self.wraps += 1
      self.start = start = 0
      end = start + self.steps
    self.start += self.steps
    return np.ravel(self.subsequences[:, start:end])
