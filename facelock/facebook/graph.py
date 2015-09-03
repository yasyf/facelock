from secrets import Secrets
from auth_code import AuthCode
from photo import Photo
from ..stores.redis_store import RedisStore
import facebook

class Graph(object):
  app_id = Secrets.get('app_id')
  version = 2.4
  access_token_store = RedisStore('facelock:facebook:graph:access_tokens')

  def __init__(self, access_token=None):
    self.access_token = access_token or Secrets.get('access_token')
    self._graph = facebook.GraphAPI(self.access_token, self.version)

  @property
  def auth_code(self):
    return AuthCode(self, self.new_auth_code())

  @property
  def fb_id(self):
    return self._graph.get_object('me')['id']

  @classmethod
  def for_user(cls, user_id=None):
    if user_id and cls.access_token_store.exists(user_id):
      access_token = cls.access_token_store.get(user_id)
    else:
      access_token = cls().auth_code.poll()
      if user_id:
        cls.access_token_store.setex(user_id, access_token, 5184000)  # 60 Days
    return Graph(access_token)

  def new_auth_code(self):
    args = {
      'client_id': self.app_id,
      'type': 'device_code',
      'scope': 'user_photos'
    }
    return self._graph.request('oauth/device', post_args=args)

  def poll_auth_code(self, code):
    args = {
      'client_id': self.app_id,
      'type': 'device_token',
      'code': code
    }
    return self._graph.request('oauth/device', post_args=args)

  def photos(self):
    id = self.fb_id
    photos = self._graph.get_connections('me', 'photos', fields='images,tags')
    after = True
    while after:
      for photo in photos['data']:
        yield Photo(photo, id)
      after = photos['paging']['cursors'].get('after')
      photos = self._graph.get_connections('me', 'photos', fields='images,tags', after=after)
    raise StopIteration



