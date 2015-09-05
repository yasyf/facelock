import os

class ConfigError(Exception):
  pass

class Config(object):
  @classmethod
  def get(cls, key):
    try:
      return open(cls.get_filename(key))
    except OSError:
      raise ConfigError(key)

  @classmethod
  def get_filename(cls, key):
    directory = os.path.dirname(__file__)
    return os.path.abspath('{current}/data/{key}'.format(current=directory, key=key))

  @classmethod
  def check_filename(cls, key):
    filename = cls.get_filename(key)
    if not os.path.exists(filename):
      raise ConfigError(key)
    return filename
