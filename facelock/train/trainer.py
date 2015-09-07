import numpy as np
from ..helpers.photo_collection import photo_collection
from .model import Model, TrainingLabel
from .processing import ProcessingMixin

class Trainer(object, ProcessingMixin):
  def __init__(self, positives=(), negatives=(), stock_negatives=(), positive_limit=None, negative_limit=None):
    self.positives = positives
    self.stock_negatives = stock_negatives
    self.negatives = negatives
    self.positive_limit = positive_limit
    self.negative_limit = negative_limit
    self.preprocess = True

  def processed(self, photos, limit=None):
    count = 0
    for photo in photos:
      if self.preprocess:
        processed = self.process(photo)
      else:
        processed = photo
      if processed is None:
        # Skip if no or multiple faces
        continue
      else:
        count += 1
        if limit and count > limit:
          raise StopIteration
        else:
          yield processed
    raise StopIteration

  @photo_collection
  def processed_positives(self):
    return self.processed(self.positives, self.positive_limit)

  @photo_collection
  def processed_stock_negatives(self):
    return self.processed(self.stock_negatives)

  @photo_collection
  def processed_negatives(self):
    return self.processed(self.negatives, self.negative_limit)

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

    for negative in self.processed_stock_negatives():
      faces.append(negative.raw())
      labels.append(TrainingLabel.NEGATIVE)

    model = Model.new_recognizer()
    model.train(np.asarray(faces), np.asarray(labels))

    return Model(model, positives=positives, negatives=negatives)
