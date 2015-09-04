from facelock.facebook.graph import Graph

USER_ID = 'YasyfM'

if __name__ == '__main__':
  graph = Graph.for_user(USER_ID)
  photos = graph.photos()
  photos.save_n(100, 'tmp/{user_id}', user_id=USER_ID)
