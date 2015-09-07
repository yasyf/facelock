import requests
import datetime
import calendar
import email
import urllib
import base64
import hmac
import hashlib
from secrets import Secrets
from . import DuoError

class Request(object):
  endpoint_prefix = 'auth/v2'
  request_type = 'GET'

  def __init__(self, endpoint, params):
    self.endpoint = endpoint
    self.params = params
    self.date = datetime.datetime.utcnow()

  @property
  def host(self):
    return Secrets.get('api_hostname')

  @property
  def formatted_date(self):
    return email.utils.formatdate(calendar.timegm(self.date))

  @property
  def path(self):
    return '/{prefix}/{endpoint}'.format(prefix=self.endpoint_prefix, endpoint=self.endpoint)

  @property
  def url(self):
    return 'https://{host}{path}'.format(host=self.host, path=self.path)

  @property
  def headers(self):
    return {'Date': self.formatted_date, 'Authorization': 'Basic {auth}'.format(auth=self.generate_auth())}

  def generate_auth(self):
    components = [self.formatted_date, self.request_type.upper(), self.host.lower(), self.path]

    args = []
    for key, val in sorted(self.params, key=lambda (k,v): k):
      args.append('{k}={v}'.format(k=urllib.quote(key, '~'), v=urllib.quote(val, '~')))
    components.append('&'.join(args))

    signature_string = '\n'.join(components)
    signature = hmac.new(Secrets.get('secret_key'), signature_string, hashlib.sha1).hexdigest()

    auth_string = '{key}:{signature}'.format(key=Secrets.get('integration_key'), signature=signature)
    return base64.b64encode(auth_string)

  def _make_request(self):
    return getattr(requests, self.request_type.lower())(self.url, self.params, headers=self.headers)

  def execute(self):
    response = self._make_request().json()
    if response['stat'] == 'OK':
      return response['response']
    else:
      raise DuoError(response)

class GetRequest(Request):
  request_type = 'GET'

class PostRequest(Request):
  request_type = 'POST'
