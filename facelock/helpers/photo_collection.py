import functools
import os
from ..helpers.dirs import mkdir_p

class PhotoCollection(object):
  def __init__(self, generator):
    self.generator = generator
    self._limit = None
    self.count = 0

  def __iter__(self):
    return self

  def next(self):
    self.count += 1
    if self._limit and self.count > self._limit:
      raise StopIteration
    return next(self.generator)

  def limit(self, i):
    self._limit = int(i)
    return self

  def save_n(self, n, path, **format_args):
    path = path.format(**format_args)
    mkdir_p(os.path.join(path, 'null.png'))
    for i in range(n):
      try:
        photo = next(self)
      except StopIteration:
        break
      photo.save('{path}/{i}.png'.format(path=path, i=i))

  def call(self, method_name):
    return (getattr(photo, method_name)() for photo in self)

def photo_collection(f):
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    return PhotoCollection(f(*args, **kwargs))

  return wrapper
