
"""Simple gap timer: counts up by gap, keeping track of number of passes"""

class gaptimer():

  def __init__(self, gap, maxupdate):
    self.gap = gap
    self.maxupdate = maxupdate
    self.previous = 0
    self.current = 0
    self.updates = 0
    self._alive = True

  def reset(self):
    self.previous = 0
    self.current = 0
    self.updates = 0
    self._alive = True

  def update(self):
    self.current += 1
    if self.current - self.previous >= self.gap:
      self.previous = self.current
      self.updates += 1
      return True
    else:
      return False

  def alive(self):
    return self.updates < self.maxupdate

