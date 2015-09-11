import os
import fnmatch
import numpy as np
from ..helpers.photo_collection import photo_collection
from .model import Model, TrainingLabel
from ..cv.photo import Photo
from ..facebook.graph import Graph
from .processing import ProcessingMixin
from ..config import Config

class Trainer(object, ProcessingMixin):
  def __init__(self, positives=(), negatives=(), stock_negatives=(), positive_limit=None, negative_limit=None,
               user_id=None):
    self.positives = positives
    self.stock_negatives = stock_negatives
    self.negatives = negatives
    self.positive_limit = positive_limit
    self.negative_limit = negative_limit
    self.user_id = user_id
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

    return Model(model, positives=positives, negatives=negatives, user_id=self.user_id)

  @staticmethod
  def stock_negative_samples():
    for sample_folder in Config.NEGATIVE_SAMPLE_FOLDERS:
      for root, _, files in os.walk(Config.check_filename(sample_folder)):
        for pattern in Config.NEGATIVE_SAMPLE_PATTERNS:
          for fn in fnmatch.filter(files, pattern):
            yield Photo.from_path(os.path.join(root, fn))
    raise StopIteration

  @staticmethod
  def negative_samples(user_id):
    for user in Config.ALL_USERS:
      if user != user_id:
        for photo in Graph.for_user(user).photos().limit(Config.NEGATIVE_N * Config.FETCHING_BUFFER):
          yield photo.to_cv()
    raise StopIteration

  @staticmethod
  def positive_samples(user_id):
    return Graph.for_user(user_id).photos().limit(Config.POSITIVE_N * Config.FETCHING_BUFFER).call('to_cv')

  @classmethod
  def default(cls, user_id):
    return Trainer(
      positives=cls.positive_samples(user_id),
      negatives=cls.negative_samples(user_id),
      stock_negatives=cls.stock_negative_samples(),
      positive_limit=Config.POSITIVE_N,
      negative_limit=Config.NEGATIVE_N,
      user_id=user_id
    )
