import os

class ConfigError(Exception):
  pass

class Config(object):
  OUTPUT_DIR = 'tmp'
  MODEL_NAME = 'model.xml'

  USER_ID = 'YasyfM'
  ALL_USERS = ('YasyfM', 'rumyasr', 'jess.li.90')

  NEGATIVE_SAMPLE_FOLDERS = ('orl_faces', 'yalefaces')
  NEGATIVE_SAMPLE_PATTERNS = ('*.png', '*.pgm')

  POSITIVE_N = 100
  NEGATIVE_N = 1
  FETCHING_BUFFER = 5
  THRESHOLD = 50

  PHOTO_PERCENT = 0.1

  FACE_WIDTH = 120.0
  FACE_HEIGHT = 140.0

  CLASSIFIER_FILES = ('haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_default.xml')

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
