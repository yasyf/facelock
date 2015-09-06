from facelock.facebook.graph import Graph
from facelock.train.trainer import Trainer
from facelock.config import Config
from facelock.cv.photo import Photo
import os
import re
import cv2
import sys

OUTPUT_DIR = 'tmp'
USER_ID = 'YasyfM'
N = 10
NEGATIVE_SAMPLE_FOLDER = 'yalefaces'
NEGATIVE_SAMPLE_REGEX = '\.png'
MODEL_NAME = 'model.xml'

def save_raw(photos):
  photos.save_n(N, '{out}/raw/{user_id}', out=OUTPUT_DIR, user_id=USER_ID)

def save_preprocessed(photos):
  processed = Trainer(photos.call('to_cv')).processed_positives()
  processed.save_n(N, '{out}/preprocessed/{user_id}', out=OUTPUT_DIR, user_id=USER_ID)

def negative_samples():
  directory = Config.check_filename(NEGATIVE_SAMPLE_FOLDER)
  files = os.listdir(directory)
  return (Photo.from_path(os.path.join(directory, fn)) for fn in files if re.search(NEGATIVE_SAMPLE_REGEX, fn))

def capture_image():
  capture = cv2.VideoCapture(0)
  _, frame = capture.read()
  capture.release()

  image = Photo(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
  return Trainer.process(image)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    raise RuntimeError('Incorrect number of arguments!')

  graph = Graph.for_user(USER_ID)
  photos = graph.photos()

  if sys.argv[1] == '--save-raw':
    save_raw(photos)
  elif sys.argv[1] == '--save-processed':
    save_preprocessed(photos)
  elif sys.argv[1] == '--train':
    trainer = Trainer(photos.limit(N).call('to_cv'), negative_samples())
    model = trainer.train()
    model.save('{out}/{model}'.format(out=OUTPUT_DIR, model=MODEL_NAME))
  elif sys.argv[1] == '--predict':
    model = Trainer.load('{out}/{model}'.format(out=OUTPUT_DIR, model=MODEL_NAME))
    image = capture_image()
    if image is None:
      raise RuntimeError('No face detected!')
    else:
      label, confidence = model.predict(image.raw())
      print 'Predicted {label} with confidence of {confidence}!'.format(label=bool(label), confidence=confidence)
      image.show()
