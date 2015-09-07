import os

class ConfigError(Exception):
  pass

class Config(object):
  # Persistence
  OUTPUT_DIR = 'tmp'
  MODEL_NAME = 'model.xml'

  # Users
  USER_ID = 'YasyfM'
  ALL_USERS = ('YasyfM', 'rumyasr', 'jess.li.90', 'niki.tubacki')

  # Stock negative samples
  NEGATIVE_SAMPLE_FOLDERS = ('orl_faces', 'yalefaces')
  NEGATIVE_SAMPLE_PATTERNS = ('*.png', '*.pgm')

  # Number of images to fetch from Facebook
  POSITIVE_N = 100
  NEGATIVE_N = 1

  # Multiple of N to limit Facebook calls to
  FETCHING_BUFFER = 5

  # Confidence threshold for positive result
  THRESHOLD = 5000

  # Percentage of photo surrounding Facebook Tag center to keep
  PHOTO_PERCENT = 0.1

  # Percentage of photo surrounding eyes to keep
  VERTICAL_EYE_OFFSET = 0.4
  HORIZONTAL_EYE_OFFSET = 0.3

  # Dimensions to resize all training images to
  FACE_WIDTH = 120.0
  FACE_HEIGHT = 140.0

  # Cascades for face detection
  FACE_CLASSIFIERS = ('haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_default.xml')
  EYEPAIR_CLASSIFIER = 'haarcascade_mcs_eyepair_big.xml'
  LEFT_EYE_CLASSIFIER = 'haarcascade_lefteye_2splits.xml'
  RIGHT_EYE_CLASSIFIER = 'haarcascade_righteye_2splits.xml'
  EYE_CLASSIFIER = 'haarcascade_eye_tree_eyeglasses.xml'

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
