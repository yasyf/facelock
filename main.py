from facelock.train.trainer import Trainer
from facelock.train.model import Model
from facelock.train.model_set import ModelSet
from facelock.config import Config
import os
import sys

class ArgumentError(Exception):
  pass

def train(user_id):
  trainer = Trainer.default(user_id)
  model = trainer.train()
  model.save()

def all_models():
  models = []
  for user_id in Config.ALL_USERS:
    path = Model.path_for_user(user_id)
    if not os.path.exists(path):
      train(user_id)
    model = Model.load_for_user(user_id)
    models.append(model)
  return models

def assert_num_args(n):
  if len(sys.argv) < (n + 2):
    raise ArgumentError('Incorrect number of arguments!')

def arg(n):
  assert_num_args(n)
  return sys.argv[n + 1]

if __name__ == '__main__':
  if arg(0) == '--train':
    try:
      train(arg(1))
    except ArgumentError:
      for user_id in Config.ALL_USERS:
        train(user_id)
  elif arg(0) == '--predict':
    model = Model.load_for_user(arg(1))
    label, confidence, images = model.run_prediction_loop(raise_on_no_face=True)

    print model.last_prediction_data
    print 'Predicted {label} with confidence of {confidence}!'.format(label=label.name, confidence=confidence)

    for image in images:
      try:
        key = chr(image.show()).upper()
      except ValueError:
        key = None
      if key == 'Y':
        print 'Recorded hit!'
      elif key == 'N':
        print 'Recorded miss!'
  elif arg(0) == '--identify':
    best_guess, confidence = ModelSet(all_models()).predict()
    if not best_guess:
      raise RuntimeError('No user could be identified!')
    print 'User is {user_id} with confidence {confidence}!'.format(user_id=best_guess, confidence=confidence)
