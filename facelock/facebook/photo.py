from ..cv.photo import Photo as CVPhoto

class Photo(object):
  def __init__(self, photo_data, user_id=None):
    self.url = photo_data['images'][0]['source']
    self.id = photo_data['id']
    self.width = float(photo_data['width'])
    self.height = float(photo_data['height'])
    self.tag = None
    if user_id:
      tags = filter(lambda t: t['id'] == user_id, photo_data['tags']['data'])
      if tags:
        self.tag = tags[0]

  @property
  def tag_x(self):
    if self.tag:
      return float(self.tag['x'])

  @property
  def tag_y(self):
    if self.tag:
      return float(self.tag['y'])

  def _crop_to_tag(self, photo):
    PHOTO_PERCENT = 0.2

    bounding_box_center_x = (self.tag_x / 100) * photo.width
    bounding_box_center_y = (self.tag_y / 100) * photo.height

    x1 = max(bounding_box_center_x - (PHOTO_PERCENT * photo.width), 0)
    y1 = max(bounding_box_center_y - (PHOTO_PERCENT * photo.height), 0)

    x2 = min(bounding_box_center_x + (PHOTO_PERCENT * photo.width), photo.width)
    y2 = min(bounding_box_center_y + (PHOTO_PERCENT * photo.height), photo.height)

    return photo.crop(x1, x2, y1, y2)

  def save(self, path):
    return self.to_cv().save(path)

  def to_cv(self):
    photo = CVPhoto.from_url(self.url)
    if self.tag:
      photo = self._crop_to_tag(photo)
    return photo

  def __repr__(self):
    return '({x}, {y}) @ {url}'.format(x=self.tag_x, y=self.tag_y, url=self.url)
