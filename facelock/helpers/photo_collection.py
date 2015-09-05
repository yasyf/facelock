import functools
import os

class PhotoCollection(object):
  def __init__(self, generator):
    self.generator = generator

  def __iter__(self):
    return self

  def next(self):
    return next(self.generator)

  def save_n(self, n, path, **format_args):
    path = path.format(**format_args)
    if not os.path.exists(path):
      os.makedirs(path)
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
