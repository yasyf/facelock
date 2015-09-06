import os

class ConfigError(Exception):
  pass

class Config(object):
  OUTPUT_DIR = 'tmp'
  USER_ID = 'YasyfM'
  ALL_USERS = ['YasyfM', 'rumyasr', 'jess.li.90']
  POSITIVE_N = 100
  NEGATIVE_N = 50
  THRESHOLD = 3000
  NEGATIVE_SAMPLE_FOLDERS = ['yalefaces', 'orl_faces']
  NEGATIVE_SAMPLE_PATTERN = '*.png'
  MODEL_NAME = 'model.xml'

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
