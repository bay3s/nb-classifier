from src.Document import Document


class NaiveBayes:

  def __init__(self):
    self.classes = 0

  def classify(self, doc: Document):
    print('Classify')
