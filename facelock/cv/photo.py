import urllib
import cv2
import numpy as np
from ..config import Config

class Photo(object):
  face_classifiers = map(lambda fn: cv2.CascadeClassifier(Config.check_filename(fn)), Config.CLASSIFIER_FILES)
  eye_classifier = cv2.CascadeClassifier(Config.check_filename(Config.EYE_CLASSIFIER))

  def __init__(self, image):
    self.image = image

  @property
  def height(self):
    return self.image.shape[0]

  @property
  def width(self):
    return self.image.shape[1]

  def crop(self, x1, x2, y1, y2):
    return self.__class__(self.image[y1:y2, x1:x2])

  def rotate(self, angle, center, scale=1.0):
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return self.__class__(cv2.warpAffine(self.image, rotation_matrix, (self.width, self.height)))

  def detect_eyes(self):
    """
    :return: (left_eye, right_eye)
    """
    eyes = self.eye_classifier.detectMultiScale(self.image)
    if len(eyes) == 2:
      if eyes[0][0] < eyes[1][0]:
        return eyes[0], eyes[1]
      else:
        return eyes[1], eyes[0]

  def detect_face(self, multi=False):
    """
    :return: (x, y, width, height)
    """
    for classifier in self.face_classifiers:
      faces = classifier.detectMultiScale(self.image, scaleFactor=1.3, minNeighbors=5, minSize=(75, 75))
      if multi and len(faces) > 0:
        return faces
      elif len(faces) == 1:
        return faces[0]

  def equalize(self):
    return self.__class__(cv2.equalizeHist(self.image))

  def resize(self, width, height):
    return self.__class__(cv2.resize(self.image, (int(width), int(height)), interpolation=cv2.INTER_LANCZOS4))

  def save(self, path):
    cv2.imwrite(path, self.image)

  def show(self):
    cv2.imshow('Photo', self.image)
    return cv2.waitKey(0)

  def raw(self):
    return self.image

  @classmethod
  def from_url(cls, url):
    request = urllib.urlopen(url)
    array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    return cls(cv2.imdecode(array, cv2.IMREAD_GRAYSCALE))

  @classmethod
  def from_path(cls, path):
    return cls(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
