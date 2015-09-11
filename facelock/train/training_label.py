from enum import IntEnum

class TrainingLabel(IntEnum):
  POSITIVE = 1
  NEGATIVE = 0
  UNKNOWN = -1
  NO_IMAGE = -1
