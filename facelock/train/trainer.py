import cv2
import numpy as np
from ..helpers.photo_collection import photo_collection

class Trainer(object):
  FACE_WIDTH = 120.0
  FACE_HEIGHT = 140.0

  def __init__(self, positives, negatives):
    self.positives = positives
    self.negatives = negatives

  def crop_to_face(self, photo, face):
    (x, y, width, height) = face
    crop_height = (width / self.FACE_WIDTH) * self.FACE_HEIGHT
    middle = y + (height / 2)
    y1 = max(0, middle - (crop_height / 2))
    y2 = min(photo.height - 1, middle + (crop_height / 2))
    return photo.crop(x, x + width, y1, y2)

  @photo_collection
  def processed_positives(self):
    for photo in self.positives:
      face = photo.detect_face()
      if face is None:
        # Skip if no or multiple faces
        continue
      yield self.crop_to_face(photo, face).resize(self.FACE_WIDTH, self.FACE_HEIGHT)
    raise StopIteration

  @photo_collection
  def processed_negatives(self):
    return (neg.resize(self.FACE_WIDTH, self.FACE_HEIGHT) for neg in self.negatives)

  def train(self):
    faces, labels = [], []
    positives, negatives = self.processed_positives(), self.processed_negatives()

    faces.extend(positives.call('raw'))
    labels.extend([1] * len(faces))

    faces.extend(negatives.call('raw'))
    labels.extend([-1] * (len(faces) - len(labels)))

    model = cv2.createEigenFaceRecognizer()
    model.train(np.asarray(faces), np.asarray(labels))

    return model
