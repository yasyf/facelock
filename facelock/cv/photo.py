import urllib
import cv2
import numpy as np

class Photo(object):
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
