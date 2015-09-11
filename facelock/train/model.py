import numpy as np
import cv2
import os
import uuid
from ..helpers.dirs import mkdir_p
from ..helpers.capture import capture_face
from ..config import Config
from .training_label import TrainingLabel
from contextlib2 import ExitStack

class PredictionData(object):
  def __init__(self, labels=None, confidences=None, images=None):
    self.labels = labels or []
    self.confidences = confidences or []
    self.images = images or []

  def __str__(self):
    return str(zip(self.labels, self.confidences))

class Model(object):
  def __init__(self, model, threshold=Config.THRESHOLD, positives=None, negatives=None, user_id=None):
    self.model = model
    self.threshold = threshold
    self.positives = positives or []
    self.negatives = negatives or []
    self.last_prediction_data = PredictionData()
    self.user_id = user_id or str(uuid.uuid4())

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
    label, _ = self.predict_multi()
    if label != TrainingLabel.POSITIVE:
      return TrainingLabel.NEGATIVE
    return label

  def predict_multi(self, min_positives=3, image_source=None, raise_on_no_face=False):
    with ExitStack() as stack:
      if image_source:
        image_source = iter(image_source)
      else:
        capture = cv2.VideoCapture(0)
        image_source = iter(lambda: capture_face(capture))
        stack.callback(capture.release)

      labels, confidences, images = [], [], []
      for i in range(min_positives):
        image = next(image_source)
        if image is None:
          if raise_on_no_face:
            raise RuntimeError('No face detected!')
          else:
            label, confidence = TrainingLabel.NO_IMAGE, 0
        else:
          label, confidence = self.evaulate(image)
          images.append(image)

        labels.append(label)
        confidences.append(confidence)

      if len(set(labels)) == 1:
        final_label = labels[0]
        final_confidence = np.mean(confidences)
      else:
        final_label = TrainingLabel.UNKNOWN
        final_confidence = 0

      self.last_prediction_data = PredictionData(labels, confidences, images)
      return final_label, final_confidence

  @staticmethod
  def _save_images(images, model_path, name):
    directory = os.path.join(os.path.dirname(model_path), name)
    mkdir_p(directory)
    for i, image in enumerate(images):
      image.save(os.path.join(directory, '{i}.png'.format(i=i)))

  def save(self, path=None):
    if not path and not self.user_id:
      raise RuntimeError('path must be provided!')
    path = path or self.path_for_user(self.user_id)
    mkdir_p(os.path.dirname(path))
    self.model.save(path)
    self._save_images(self.positives, path, 'positives')
    self._save_images(self.negatives, path, 'negatives')

  @classmethod
  def load(cls, path, **kwargs):
    model = cls.new_recognizer()
    model.load(path)
    return cls(model, **kwargs)

  @staticmethod
  def path_for_user(user_id):
    return '{out}/model/{user_id}/{model}'.format(out=Config.OUTPUT_DIR, user_id=user_id, model=Config.MODEL_NAME)

  @classmethod
  def load_for_user(cls, user_id, **kwargs):
    return cls.load(cls.path_for_user(user_id), user_id=user_id, **kwargs)

  @staticmethod
  def new_recognizer():
    return cv2.createEigenFaceRecognizer()
