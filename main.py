from facelock.facebook.graph import Graph

if __name__ == '__main__':
  graph = Graph.for_user('YasyfM')
  photos = graph.photos()
  for i in range(100):
    next(photos).to_cv().save('tmp/YasyfM/{i}.png'.format(i=i))
