import os

def mkdir_p(path):
  if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
