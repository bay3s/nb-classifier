import csv
from copy import deepcopy

class NaiveBayes:
  
  def __init__(self, num_categories: int, vocab_file_location: str, training_data_location: str, training_labels_location: str, test_data_location: str, test_labels_location: str):
    self.vocab_file_location = vocab_file_location

    self.training_data_location = training_data_location
    self.training_labels_location = training_labels_location

    self.test_data_location = test_data_location
    self.test_labels_location = test_labels_location
    
    self.vocabulary = dict()

    self.training_labels = dict()
    self.training_label_counts = dict()
    
    self.priors = dict()
    self.joint_counts = dict()
    self.posteriors = dict()

    self.test_labels = dict()
    
    self.smoothing_constant = 1
    self.num_categories = num_categories
  
  def train(self):
    self.init_vocabulary()

    self.init_training_labels()
    self.init_priors()
    self.init_posteriors()

  def init_training_labels(self, refresh = True) -> bool:
    if refresh:
      self.training_labels.clear()
      self.training_label_counts.clear()

    reader = csv.reader(open(self.training_labels_location))
  
    for index, row in enumerate(reader):
      label = int(row[0])
      self.training_labels[index + 1] = label

      if label not in self.training_label_counts.keys():
        self.training_label_counts[label] = 0

      self.training_label_counts[label] += 1

    return True

  def init_priors(self) -> bool:
    for label in self.training_label_counts.keys():
        self.priors[label] = float(self.training_label_counts[label] / len(self.training_labels))

    return True

  def init_test_labels(self, refresh = True) -> bool:
    if refresh:
      self.test_labels.clear()

    reader = csv.reader(open(self.test_data_location))
    for index, row in enumerate(reader):
      label = int(row[0])
      self.test_labels[index + 1] = label

    return True

  def init_vocabulary(self) -> bool:
    reader = csv.reader(open(self.vocab_file_location))
    self.vocabulary.clear()

    for index, row in enumerate(reader):
        word = row[0]
        self.vocabulary[index + 1] = word

    return True

  def get_posteriors_key(self, word_id: int, label: int):
    return f'{word_id} & {label}'

  def init_posteriors(self):
    training_data_reader = csv.reader(open(self.training_data_location))
    training_data = list(enumerate(training_data_reader))
    training_labels = list(self.training_labels.values())
    joint_counts = dict()

    for index, data in enumerate(training_data):
      row_data = data[1][0].split()
      doc = int(row_data[0])
      word = int(row_data[1])
      count = int(row_data[2])

      # adjusting for index
      label = int(training_labels[doc - 1])
      key = self.get_posteriors_key(word, label)

      if key not in self.joint_counts.keys():
        joint_counts[key] = count
      joint_counts[key] += count
  
    for key, count in joint_counts.items():
      label = int(key.split()[2])
      self.posteriors[key] = (count + self.smoothing_constant) / (self.training_label_counts[label] + self.smoothing_constant * self.num_categories)

  def run_classifier(self) -> dict:
    test_data_reader = csv.reader(open(self.test_data_location))
    test_data = list(enumerate(test_data_reader))
    likelihood_estimates = dict()

    for index, data in enumerate(test_data):
      row_data = data[1][0].split()
      doc = int(row_data[0])
      word = int(row_data[1])

      if doc not in likelihood_estimates.keys():
        likelihood_estimates[doc] = deepcopy(self.priors)

      for label in likelihood_estimates[doc].keys():
        posteriors_key = self.get_posteriors_key(word, label)

        if posteriors_key in self.posteriors.keys():
          likelihood_estimates[doc][label] = likelihood_estimates[doc][label] * self.posteriors[posteriors_key]
        else:
          likelihood_estimates[doc][label] = likelihood_estimates[doc][label] * self.smoothing_constant / (self.training_label_counts[label] + self.smoothing_constant * self.num_categories)

    results = dict()
    for doc in likelihood_estimates.keys():
      max_probability = max(likelihood_estimates[doc], key = lambda k: likelihood_estimates[doc].get(k))
      results[doc] = max_probability

    return results
