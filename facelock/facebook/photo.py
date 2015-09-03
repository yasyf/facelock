class Photo(object):
  def __init__(self, photo_data, user_id=None):
    self.url = photo_data['images'][0]['source']
    self.tag = None
    if user_id:
      tags = filter(lambda t: t['id'] == user_id, photo_data['tags']['data'])
      if tags:
        self.tag = tags[0]

  @property
  def tag_x(self):
    if self.tag:
      return self.tag['x']

  @property
  def tag_y(self):
    if self.tag:
      return self.tag['y']

  def __repr__(self):
    return '({x}, {y}) @ {url}'.format(x=self.tag_x, y=self.tag_y, url=self.url)
