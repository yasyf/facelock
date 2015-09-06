from facelock.facebook.graph import Graph
from facelock.train.trainer import Trainer
from facelock.train.model import Model
from facelock.config import Config
from facelock.cv.photo import Photo
from facelock.helpers.dirs import mkdir_p
import os
import cv2
import sys
import fnmatch

OUTPUT_DIR = 'tmp'
USER_ID = 'YasyfM'
ALL_USERS = ['YasyfM', 'rumyasr', 'jess.li.90']
POSITIVE_N = 100
NEGATIVE_N = 50
THRESHOLD = 3000
NEGATIVE_SAMPLE_FOLDERS = ['yalefaces', 'orl_faces']
NEGATIVE_SAMPLE_PATTERN = '*.png'
MODEL_NAME = 'model.xml'

def save_raw(photos):
  photos.save_n(POSITIVE_N, '{out}/raw/{user_id}', out=OUTPUT_DIR, user_id=USER_ID)

def save_preprocessed(photos):
  processed = Trainer(photos.call('to_cv')).processed_positives()
  processed.save_n(POSITIVE_N, '{out}/preprocessed/{user_id}', out=OUTPUT_DIR, user_id=USER_ID)

def negative_samples():
  for sample_folder in NEGATIVE_SAMPLE_FOLDERS:
    for root, _, files in os.walk(Config.check_filename(sample_folder)):
      for fn in fnmatch.filter(files, NEGATIVE_SAMPLE_PATTERN):
        yield Photo.from_path(os.path.join(root, fn))
  for user in ALL_USERS:
    if user != USER_ID:
      for photo in Graph.for_user(user).photos().limit(NEGATIVE_N):
        yield photo.to_cv()
  raise StopIteration

def capture_image():
  capture = cv2.VideoCapture(0)
  while True:
    _, frame = capture.read()
    image = Photo(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    processed = Trainer.process(image)
    if processed is not None:
      capture.release()
      return processed

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
    trainer = Trainer(photos.limit(POSITIVE_N).call('to_cv'), negative_samples())
    model = trainer.train()
    path = '{out}/model/{user_id}/{model}'.format(out=OUTPUT_DIR, user_id=USER_ID, model=MODEL_NAME)
    mkdir_p(path)
    model.save(path)
  elif sys.argv[1] == '--predict':
    model = Model.load('{out}/model/{user_id}/{model}'.format(out=OUTPUT_DIR, user_id=USER_ID, model=MODEL_NAME))
    model.threshold = THRESHOLD
    image = capture_image()
    if image is None:
      raise RuntimeError('No face detected!')
    else:
      label, confidence = model.predict(image)
      print 'Predicted {label} with confidence of {confidence}!'.format(label=label.name, confidence=confidence)
      image.show()
