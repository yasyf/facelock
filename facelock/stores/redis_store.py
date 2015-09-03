from redis import StrictRedis

class RedisStore(object):
  redis = StrictRedis()

  def __init__(self, namespace):
    self.namespace = namespace

  def key(self, name):
    return '{namespace}:{name}'.format(namespace=self.namespace, name=name)

  def exists(self, name):
    return self.redis.exists(self.key(name))

  def setex(self, name, value, time):
    return self.redis.set(self.key(name), value, time)

  def set(self, name, value):
    self.setex(name, value, None)

  def get(self, name):
    return self.redis.get(self.key(name))
