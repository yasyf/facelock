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

def save_raw(photos):
  photos.save_n(Config.POSITIVE_N, '{out}/raw/{user_id}', out=Config.OUTPUT_DIR, user_id=Config.USER_ID)

def save_preprocessed(photos):
  processed = Trainer(photos.call('to_cv')).processed_positives()
  processed.save_n(Config.POSITIVE_N, '{out}/preprocessed/{user_id}', out=Config.OUTPUT_DIR, user_id=Config.USER_ID)

def negative_samples():
  for sample_folder in Config.NEGATIVE_SAMPLE_FOLDERS:
    for root, _, files in os.walk(Config.check_filename(sample_folder)):
      for fn in fnmatch.filter(files, Config.NEGATIVE_SAMPLE_PATTERN):
        yield Photo.from_path(os.path.join(root, fn))
  for user in Config.ALL_USERS:
    if user != Config.USER_ID:
      for photo in Graph.for_user(user).photos().limit(Config.NEGATIVE_N):
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

  graph = Graph.for_user(Config.USER_ID)
  photos = graph.photos()

  if sys.argv[1] == '--save-raw':
    save_raw(photos)
  elif sys.argv[1] == '--save-processed':
    save_preprocessed(photos)
  elif sys.argv[1] == '--train':
    trainer = Trainer(photos.limit(Config.POSITIVE_N).call('to_cv'), negative_samples())
    model = trainer.train()
    path = '{out}/model/{user_id}/{model}'.format(out=Config.OUTPUT_DIR,
                                                  user_id=Config.USER_ID, model=Config.MODEL_NAME)
    mkdir_p(path)
    model.save(path)
  elif sys.argv[1] == '--predict':
    model = Model.load('{out}/model/{user_id}/{model}'.format(out=Config.OUTPUT_DIR,
                                                              user_id=Config.USER_ID, model=Config.MODEL_NAME))
    model.threshold = Config.THRESHOLD
    image = capture_image()
    if image is None:
      raise RuntimeError('No face detected!')
    else:
      label, confidence = model.predict(image)
      print 'Predicted {label} with confidence of {confidence}!'.format(label=label.name, confidence=confidence)
      image.show()
