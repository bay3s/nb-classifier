import pandas as pd


class NaiveBayes:

  def __init__(
    self,
    vocab_path: str,
    labels_path: str,
    train_data_path: str,
    train_labels_path: str,
    test_data_path: str,
    test_labels_path: str
  ):
    self.words = NaiveBayes.read_csv(vocab_path, columns = ['word'])
    self.labels = NaiveBayes.read_csv(labels_path, columns = ['name'])
 
    self.training_data = NaiveBayes.read_csv(train_data_path, columns = ['doc.id', 'word.id', 'count'])
    self.training_data = self.training_data[['doc.id', 'word.id', 'count']].apply(pd.to_numeric)
    
    self.training_labels = NaiveBayes.read_csv(train_labels_path, columns = ['label.id'])
    self.training_labels = self.training_labels[['label.id']].apply(pd.to_numeric)
    self.training_labels['doc.id'] = self.training_labels.index + 1
    self.training_labels.set_index('doc.id')

    self.test_data = NaiveBayes.read_csv(test_data_path, columns = ['doc.id', 'word.id', 'count'])
    self.test_data = self.test_data[['doc.id', 'word.id', 'count']].apply(pd.to_numeric)

    self.test_labels = NaiveBayes.read_csv(test_labels_path, columns = ['label.id'])
    self.test_labels = self.test_labels[['label.id']].apply(pd.to_numeric)
    
    # P(Y)
    self.priors = None
    
    # P(X | Y)
    self.maximum_posteriori = None
    self.smoothing = 1.0
  
  def train(self):
    self.init_priors()
    self.init_maximum_posteriori()

  def init_maximum_posteriori(self):
    self.maximum_posteriori = pd.DataFrame(columns = ['word.id', 'label.id', 'estimate'])

    for label_idx, label in self.labels.iterrows():
      label_id = label['id']
      documents = self.training_labels[self.training_labels['label.id'] == label_id]
      document_ids = documents['doc.id'].tolist()

      filtered_data = self.training_data.loc[self.training_data['doc.id'].isin(document_ids)]
      filtered_data.drop('doc.id', axis = 1, inplace = True)
      aggregated = filtered_data.groupby('word.id').agg({'count': 'sum'})

      maximum_posteriori = pd.DataFrame({
        'word.id': aggregated.index.values,
        'label.id': [label_id] * len(aggregated),
        'estimate': (aggregated['count'] + self.smoothing) / (aggregated['count'].sum() + len(self.words) * self.smoothing)
      })

      self.maximum_posteriori = self.maximum_posteriori.append(maximum_posteriori)

  def init_priors(self):
    training_size = self.training_labels.nunique()['label.id']

    self.priors = pd.DataFrame(columns = ['label.id', 'prior'])

    self.priors['label.id'] = self.training_labels['label.id'].unique()
    self.priors['prior'] = 0.0

    label_counts = dict()
    for idx, row in self.training_labels.iterrows():
      label = row['label.id']

      if label not in label_counts.keys():
        label_counts[label] = len(self.training_labels[self.training_labels['label.id'] == label])

      self.priors.loc[self.priors['label.id'] == label, 'prior'] = label_counts[label] / training_size

    pass

  def test(self):
    documents = self.test_data['doc.id'].unique()
    label_ids = self.labels['id'].unique()
    predictions = list()

    for doc_id in documents:
      word_ids = self.test_data[self.test_data['doc.id'] == doc_id]['word.id']
      word_count = self.test_data[self.test_data['doc.id'] == doc_id]['count'].sum()
      max_probability = 0.0
      predicted_label = None

      for label_id in label_ids:
        maximum_posteriori = self.maximum_posteriori[self.maximum_posteriori['label.id'] == label_id & self.maximum_posteriori['word.id'].isin(word_ids)]
        likelihood = 0.0

        for word_id in word_ids:
          if maximum_posteriori[maximum_posteriori['word.id'] == word_id]['word.id'].any():
            estimate = maximum_posteriori.loc[maximum_posteriori['word.id'] == word_id, 'estimate'].iloc[0]
            if likelihood > 0.0:
              likelihood = likelihood * estimate
            else:
              likelihood = estimate
          else:
            estimate = 1 / (word_count + len(self.words))
            if likelihood > 0.0:
              likelihood = likelihood * estimate
            else:
              likelihood = estimate

        prior = self.priors.loc[self.priors['label.id'] == label_id, 'prior'].iloc[0]
        current_probability = likelihood * prior
        if current_probability > max_probability:
          predicted_label = label_id
          max_probability = current_probability

      predictions.append(predicted_label)

    return predictions

  @staticmethod
  def read_csv(file: str, columns: list, converters: dict = {}):
    df = pd.read_csv(file, names = columns, sep = ' ', converters = converters)
    df.insert(0, 'id', df.index + 1)
    df.set_index('id')

    return df
