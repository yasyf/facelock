from facelock.facebook.graph import Graph
from facelock.train.trainer import Trainer

USER_ID = 'YasyfM'
N = 10

def save_raw(photos):
  photos.save_n(N, 'tmp/raw/{user_id}', user_id=USER_ID)

def save_preprocessed(photos):
  Trainer(photos.call('to_cv')).processed().save_n(N, 'tmp/preprocessed/{user_id}', user_id=USER_ID)

if __name__ == '__main__':
  graph = Graph.for_user(USER_ID)
  photos = graph.photos()
  # save_raw(photos)
  save_preprocessed(photos)
