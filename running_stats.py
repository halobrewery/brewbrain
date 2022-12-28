import numpy as np 
import math

class RunningStats(object):
  """Computes running mean and standard deviation
  Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
  Adapted from:
      *
      <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
      * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
      * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>
      
  Usage:
      rs = RunningStats()
      for i in range(10):
          rs.add(np.random.randn())
          print(rs)
      print(rs.mean, rs.std)
  """

  def __init__(self, n=0., m=None, s=None):
    self.n = n
    self.m = m
    self.s = s

  def clear(self):
    self.n = 0.0
    self.m = None
    self.s = None

  def push(self, x, per_dim=False):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
      
    if per_dim:
      self.update_params(x)
    else:
      for el in x.flatten():
        self.update_params(el)

  def update_params(self, x):
    self.n += 1
    if self.n == 1:
      self.m = x
      self.s = 0.0
    else:
      prev_m = self.m.copy()
      self.m += (x - self.m) / self.n
      self.s += (x - prev_m) * (x - self.m)
        
  def add(self, other):
    if isinstance(other, RunningStats):
      sum_ns = self.n + other.n
      prod_ns = self.n * other.n
      delta2 = (other.m - self.m) ** 2.
      return RunningStats(sum_ns,
        (self.m * self.n + other.m * other.n) / sum_ns,
        self.s + other.s + delta2 * prod_ns / sum_ns
      )
    else:
      self.push(other)
      return self

  def mean(self):
    return self.m if self.n else 0.0
  def var(self):
    return self.s / (self.n) if self.n else 0.0
  def std(self):
    return np.sqrt(self.var())
      
  def __repr__(self):
    return '<RunningMean(mean={: 2.4f}, std={: 2.4f}, n={: 2f}, m={: 2.4f}, s={: 2.4f})>'.format(self.mean(), self.std(), self.n, self.m, self.s)
  def __str__(self):
    return 'mean={: 2.4f}, std={: 2.4f}'.format(self.mean(), self.std())