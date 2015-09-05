import urllib
import cv2
import numpy as np
from ..config import Config

class Photo(object):
  CLASSIFIER_FILE = 'haarcascade_frontalface_alt.xml'
  face_classifier = cv2.CascadeClassifier(Config.check_filename(CLASSIFIER_FILE))

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

  def detect_face(self, multi=False):
    """
    :return: (x, y, width, height)
    """
    faces = self.face_classifier.detectMultiScale(
      self.image,
      scaleFactor=1.3,
      minNeighbors=4,
      minSize=(30, 30),
      flags=cv2.CASCADE_SCALE_IMAGE
    )
    if multi:
      return faces
    if len(faces) == 1:
      return faces[0]

  def save(self, path):
    cv2.imwrite(path, self.image)

  def show(self):
    cv2.imshow('Photo', self.image)
    cv2.waitKey(0)

  @classmethod
  def from_url(cls, url):
    request = urllib.urlopen(url)
    array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    return cls(cv2.imdecode(array, cv2.IMREAD_GRAYSCALE))

  @classmethod
  def from_path(cls, path):
    return cls(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
