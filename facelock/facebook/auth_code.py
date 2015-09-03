import time
from facebook import GraphAPIError
from ..helpers.timeout import timeout, TimeoutError

class AuthCode(object):
  def __init__(self, graph, data):
    self.graph = graph
    self.code = data['user_code']
    self.poll_code = data['code']
    self.verification_uri = data['verification_uri']
    self.interval = data['interval']
    self.expires_in = data['expires_in']

  def _poll(self):
    while True:
      try:
        response = self.graph.poll_auth_code(self.poll_code)
      except GraphAPIError as e:
        print(e)
        if e.message != 'authorization_pending':
          raise e
      else:
        return response.get('access_token')
      time.sleep(self.interval + 1)

  def poll(self):
    print('Visit {verification_uri} and enter {code}.'.format(**self.__dict__))
    try:
      with timeout(self.expires_in):
        return self._poll()
    except TimeoutError:
      return None

