import cv2
import os
from ..helpers.dirs import mkdir_p
from enum import IntEnum

class TrainingLabel(IntEnum):
  POSITIVE = 1
  NEGATIVE = 0
  UNKNOWN = -1

class Model(object):
  def __init__(self, model, threshold=None, positives=None, negatives=None):
    self.model = model
    self.threshold = threshold
    self.positives = positives or []
    self.negatives = negatives or []

  def predict(self, image):
    label, confidence = self.model.predict(image.raw())
    if label == TrainingLabel.POSITIVE:
      if not self.threshold or confidence > self.threshold:
        return TrainingLabel.POSITIVE, confidence
      else:
        return TrainingLabel.NEGATIVE, -confidence
    else:
      return TrainingLabel.NEGATIVE, confidence

  @staticmethod
  def _save_images(images, model_path, name):
    directory = os.path.join(os.path.dirname(model_path), name)
    mkdir_p(directory)
    for i, image in enumerate(images):
      image.save(os.path.join(directory, '{i}.png'.format(i=i)))

  def save(self, path):
    mkdir_p(os.path.dirname(path))
    self.model.save(path)
    self._save_images(self.positives, path, 'positives')
    self._save_images(self.negatives, path, 'negatives')

  @classmethod
  def load(cls, path):
    model = cls.new_recognizer()
    model.load(path)
    return cls(model)

  @staticmethod
  def new_recognizer():
    return cv2.createEigenFaceRecognizer()
