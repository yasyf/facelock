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
  NO_IMAGE = -1

class PredictionData(object):
  def __init__(self, labels=None, confidences=None, images=None):
    self.labels = labels or []
    self.confidences = confidences or []
    self.images = images or []

  def __str__(self):
    return str(zip(self.labels, self.confidences))

class Model(object):
  def __init__(self, model, threshold=None, positives=None, negatives=None):
    self.model = model
    self.threshold = threshold
    self.positives = positives or []
    self.negatives = negatives or []
    self.last_prediction_data = PredictionData()

  def evaulate(self, image):
    label, confidence = self.model.predict(image.raw())
    if label == TrainingLabel.POSITIVE:
      if not self.threshold or confidence > self.threshold:
        return TrainingLabel.POSITIVE, confidence
      else:
        return TrainingLabel.NEGATIVE, -confidence
    else:
      return TrainingLabel.NEGATIVE, confidence

  def predict(self):
    label, _, _ = self.run_prediction_loop()
    if label != TrainingLabel.POSITIVE:
      return TrainingLabel.NEGATIVE
    return label

  def run_prediction_loop(self, capture=None, min_positives=3, raise_on_no_face=False):
    capture_to_use = capture or cv2.VideoCapture(0)
    images, labels, confidences = [], [], []
    for i in range(min_positives):
      image = capture_face(capture_to_use)
      if image is None:
        if raise_on_no_face:
          raise RuntimeError('No face detected!')
        else:
          label, confidence = TrainingLabel.NO_IMAGE, 0
      else:
        images.append(image)
        label, confidence = self.evaulate(image)

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

    self.last_prediction_data = PredictionData(labels, confidences, images)
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
