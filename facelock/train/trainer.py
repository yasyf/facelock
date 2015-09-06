import numpy as np
from ..helpers.photo_collection import photo_collection
from .model import Model, TrainingLabel
from ..config import Config

class Trainer(object):
  def __init__(self, positives=(), negatives=()):
    self.positives = positives
    self.negatives = negatives
    self.preprocess = True

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

  def processed(self, photos):
    for photo in photos:
      if self.preprocess:
        processed = self.process(photo)
      else:
        processed = photo
      if processed is None:
        # Skip if no or multiple faces
        continue
      else:
        yield processed
    raise StopIteration

  @photo_collection
  def processed_positives(self):
    return self.processed(self.positives)

  @photo_collection
  def processed_negatives(self):
    return self.processed(self.negatives)

  def train(self):
    faces, labels = [], []
    positives, negatives = [], []

    for positive in self.processed_positives():
      faces.append(positive.raw())
      labels.append(TrainingLabel.POSITIVE)
      positives.append(positive)

    for negative in self.processed_negatives():
      faces.append(negative.raw())
      labels.append(TrainingLabel.NEGATIVE)
      negatives.append(negative)

    model = Model.new_recognizer()
    model.train(np.asarray(faces), np.asarray(labels))

    return Model(model, positives=positives, negatives=negatives)
