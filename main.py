from facelock.facebook.graph import Graph
from facelock.train.trainer import Trainer
from facelock.config import Config
from facelock.cv.photo import Photo
import os
import re

USER_ID = 'YasyfM'
N = 10
NEGATIVE_SAMPLE_FOLDER = 'yalefaces/'
NEGATIVE_SAMPLE_REGEX = '\.png'

def save_raw(photos):
  photos.save_n(N, 'tmp/raw/{user_id}', user_id=USER_ID)

def save_preprocessed(photos):
  Trainer(photos.call('to_cv'), []).processed_positives().save_n(N, 'tmp/preprocessed/{user_id}', user_id=USER_ID)

def negative_samples():
  directory = Config.check_filename(NEGATIVE_SAMPLE_FOLDER)
  files = os.listdir(directory)
  return (Photo.from_path(os.path.join(directory, fn)) for fn in files if re.search(NEGATIVE_SAMPLE_REGEX, fn))

if __name__ == '__main__':
  graph = Graph.for_user(USER_ID)
  photos = graph.photos()
  # save_raw(photos)
  # save_preprocessed(photos)
  trainer = Trainer(photos.limit(N).call('to_cv'), negative_samples())
  model = trainer.train()
  model.save('tmp/model.xml')
