import cv2
from enum import IntEnum

class TrainingLabel(IntEnum):
  POSITIVE = 1
  NEGATIVE = 0
  UNKNOWN = -1

class Model(object):
  def __init__(self, model, threshold=None):
    self.model = model
    self.threshold = threshold

  def predict(self, image):
    label, confidence = self.model.predict(image.raw())
    if label == TrainingLabel.POSITIVE:
      if not self.threshold or confidence > self.threshold:
        return TrainingLabel.POSITIVE, confidence
      else:
        return TrainingLabel.NEGATIVE, -confidence
    else:
      return TrainingLabel.NEGATIVE, confidence

  def save(self, path):
    self.model.save(path)

  @classmethod
  def load(cls, path):
    model = cv2.createEigenFaceRecognizer()
    model.load(path)
    return cls(model)
