from ..helpers.photo_collection import photo_collection

class Trainer(object):
  FACE_WIDTH = 120.0
  FACE_HEIGHT = 140.0

  def __init__(self, photos):
    self.photos = photos

  def crop_to_face(self, photo, face):
    (x, y, width, height) = face
    crop_height = (width / self.FACE_WIDTH) * self.FACE_HEIGHT
    middle = y + (height / 2)
    y1 = max(0, middle - (crop_height / 2))
    y2 = min(photo.height - 1, middle + (crop_height / 2))
    return photo.crop(x, x + width, y1, y2)

  @photo_collection
  def processed(self):
    for photo in self.photos:
      face = photo.detect_face()
      if face is None:
        # Skip if no or multiple faces
        continue
      yield self.crop_to_face(photo, face)
    raise StopIteration
