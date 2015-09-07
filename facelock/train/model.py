import numpy as np
import cv2
import os
from ..helpers.dirs import mkdir_p
from ..helpers.capture import capture_face
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

  def run_prediction_loop(self, capture=None, min_positives=3, raise_on_no_face=False):
    capture_to_use = capture or cv2.VideoCapture(0)
    images, labels, confidences = [], [], []
    for i in range(min_positives):
      image = capture_face(capture_to_use)
      if image is None:
        if raise_on_no_face:
          raise RuntimeError('No face detected!')
        else:
          label, confidence = TrainingLabel.UNKNOWN, 0
      else:
        images.append(image)
        label, confidence = self.predict(image)

      labels.append(label)
      confidences.append(confidence)

    if capture is None:
      capture_to_use.release()

    if len(set(labels)) == 1:
      final_label = labels[0]
      final_confidence = np.mean(confidences)
    else:
      final_label = TrainingLabel.UNKNOWN
      final_confidence = 0

    print labels, confidences
    return final_label, final_confidence, images

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
