import numpy as np
import math
from ..helpers.photo_collection import photo_collection
from .model import Model, TrainingLabel
from ..config import Config

class Trainer(object):
  def __init__(self, positives=(), negatives=(), stock_negatives=(), positive_limit=None, negative_limit=None):
    self.positives = positives
    self.stock_negatives = stock_negatives
    self.negatives = negatives
    self.positive_limit = positive_limit
    self.negative_limit = negative_limit
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
  def rotate_and_crop_around_eyes(cls, photo, left_eye, right_eye):
    horizontal_offset = Config.HORIZONTAL_EYE_OFFSET * Config.FACE_WIDTH
    vertical_offset = Config.VERTICAL_EYE_OFFSET * Config.FACE_HEIGHT

    distance = np.linalg.norm(right_eye[:2] - left_eye[:2])
    reference_width = Config.FACE_WIDTH - (2 * horizontal_offset)
    scale = distance / reference_width

    direction = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    rotation = math.atan2(float(direction[1]), float(direction[0]))
    photo = photo.rotate(math.degrees(rotation), tuple(left_eye[:2]))

    x1 = max(0, left_eye[0] - (scale * horizontal_offset))
    x2 = min(photo.width, x1 + (scale * Config.FACE_WIDTH))
    y1 = max(0, left_eye[1] - (scale * vertical_offset))
    y2 = min(photo.height, y1 + (scale * Config.FACE_HEIGHT))
    return photo.crop(x1, x2, y1, y2)

  @classmethod
  def process(cls, photo):
    face = photo.detect_face()
    if face is None:
      return None
    else:
      photo = cls.crop_to_face(photo, face)

    eyes = photo.detect_eyes()
    if eyes is not None:
      photo = cls.rotate_and_crop_around_eyes(photo, *eyes)

    return photo.resize(Config.FACE_WIDTH, Config.FACE_HEIGHT).equalize()

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
