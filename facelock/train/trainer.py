import cv2
import numpy as np
from ..helpers.photo_collection import photo_collection
from .model import Model, TrainingLabel
from ..config import Config

class Trainer(object):
  def __init__(self, positives=(), negatives=()):
    self.positives = positives
    self.negatives = negatives

  @classmethod
  def crop_to_face(cls, photo, face):
    (x, y, width, height) = face
    crop_height = (width / Config.FACE_WIDTH) * Config.FACE_HEIGHT
    middle = y + (height / 2)
    y1 = max(0, middle - (crop_height / 2))
    y2 = min(photo.height - 1, middle + (crop_height / 2))
    return photo.crop(x, x + width, y1, y2)

  @classmethod
  def process(cls, photo):
    face = photo.detect_face()
    if face is None:
      return None
    return cls.crop_to_face(photo, face).resize(Config.FACE_WIDTH, Config.FACE_HEIGHT)

  @photo_collection
  def processed_positives(self):
    for photo in self.positives:
      processed = self.process(photo)
      if processed is None:
        # Skip if no or multiple faces
        continue
      else:
        yield processed
    raise StopIteration

  @photo_collection
  def processed_negatives(self):
    return (neg.resize(Config.FACE_WIDTH, Config.FACE_HEIGHT) for neg in self.negatives)

  def train(self):
    faces, labels = [], []
    positives, negatives = self.processed_positives(), self.processed_negatives()

    faces.extend(positives.call('raw'))
    labels.extend([TrainingLabel.POSITIVE] * len(faces))

    faces.extend(negatives.call('raw'))
    labels.extend([TrainingLabel.NEGATIVE] * (len(faces) - len(labels)))

    model = cv2.createEigenFaceRecognizer()
    model.train(np.asarray(faces), np.asarray(labels))

    return Model(model)
