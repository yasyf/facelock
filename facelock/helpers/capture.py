import cv2
from ..cv.photo import Photo
from ..train.processing import ProcessingMixin

def capture_face(capture):
  while True:
    _, frame = capture.read()
    image = Photo(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    processed = ProcessingMixin.process(image)
    if processed is not None:
      return processed
