import os

class Secrets(object):
  @staticmethod
  def get(key):
    return os.getenv('FACEBOOK_' + key.upper())
