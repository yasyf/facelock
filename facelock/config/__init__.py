import os

class ConfigError(Exception):
  pass

class Config(object):
  OUTPUT_DIR = 'tmp'
  MODEL_NAME = 'model.xml'

  USER_ID = 'YasyfM'
  ALL_USERS = ['YasyfM', 'rumyasr', 'jess.li.90']

  NEGATIVE_SAMPLE_FOLDERS = ['orl_faces', 'yalefaces']
  NEGATIVE_SAMPLE_PATTERN = '*.png'

  POSITIVE_N = 100
  NEGATIVE_N = 20
  THRESHOLD = 3500

  PHOTO_PERCENT = 0.15

  FACE_WIDTH = 92.0
  FACE_HEIGHT = 112.0

  CLASSIFIER_FILES = ['haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_default.xml']

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
