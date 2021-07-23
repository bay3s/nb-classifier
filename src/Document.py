class Document:

  def __init__(self, path: str):
    self.path = path
    self.contents = self.read()

  def read(self):
    return 'contents'