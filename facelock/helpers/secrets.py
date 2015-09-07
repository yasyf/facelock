import os

class BaseSecrets(object):
  PREFIX = ''

  @classmethod
  def get(cls, key):
    return os.getenv(cls.PREFIX + '_' + key.upper())
