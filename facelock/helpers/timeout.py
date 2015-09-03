import signal

class TimeoutError(Exception):
  pass

class timeout:
  def __init__(self, seconds, message=None):
    self.seconds = seconds
    self.message = message

  def __handle_timeout(self, signum, frame):
    raise TimeoutError(self.message)

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.__handle_timeout)
    signal.alarm(self.seconds)

  def __exit__(self, exc_type, exc_val, exc_tb):
    signal.alarm(0)
