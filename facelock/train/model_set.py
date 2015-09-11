from contextlib2 import ExitStack
from .training_label import TrainingLabel
from ..helpers.capture import capture_face
import cv2

class ModelSet(object):
  def __init__(self, models):
    self.models = models

  def predict(self, min_positives=3, capture=None):
    with ExitStack() as stack:
      if not capture:
        capture = cv2.VideoCapture(0)
        stack.callback(capture.release)

      images = [capture_face(capture) for _ in range(min_positives)]
      results = {model.user_id: model.predict_multi(min_positives, images) for model in self.models}
      print results
      positives = filter(lambda user_id: results[user_id][0] == TrainingLabel.POSITIVE, results.keys())
      if not positives:
        return None, 0
      best = max(positives, key=lambda user_id: results[user_id][1])
      return best, results[best][1]
