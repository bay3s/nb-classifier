from src.NaiveBayes import NaiveBayes
from src.Document import Document
import csv
import pathlib


if __name__ == '__main__':
    # build vocab
    vocab_file = open(f'{pathlib.Path(__file__).parent.resolve()}/data/vocabulary.txt')
    vocab_reader = csv.reader(vocab_file)
    vocab = dict()

    word_id = 1
    for row in vocab_reader:
        word = row[0]
        vocab[word_id] = word
        word_id += 1

    # calculate the probabilities for each labels
    training_label_file = open(f'{pathlib.Path(__file__).parent.resolve()}/data/train.label')
    training_label_reader = csv.reader(training_label_file)

    probability_labels = dict()
    count_labels = dict()
    document_labels = dict()
    num_rows = doc_id = 0

    for row in training_label_reader:
        doc_id += 1
        num_rows = doc_id

        label = int(row[0])
        document_labels[doc_id] = label

        if label not in count_labels.keys():
            count_labels[label] = 1
        else:
            count_labels[label] += 1

    for label in count_labels.keys():
        probability_labels[label] = float(count_labels[label] / num_rows)

    joint_distributions = dict()
    maximum_posteriori = dict()

    training_data_file = open(f'{pathlib.Path(__file__).parent.resolve()}/data/train.data')
    training_data_reader = csv.reader(training_data_file)
    J = 0.5

    for data in training_data_reader:
        num_rows += 1
        row_data = data[0].split()

        doc_id = int(row_data[0])
        word_id = int(row_data[1])
        word_count = int(row_data[2])

        if word_id not in joint_distributions.keys():
            joint_distributions[word_id] = dict()

        document_label = document_labels[doc_id]
        joint_distributions[word_id][document_label] = word_count
    
        if word_id not in maximum_posteriori.keys():
            maximum_posteriori[word_id] = dict()

        # maximum a posteriori with Laplace smoothing. Prior distributed equally across labels.
        maximum_posteriori[word_id][document_label] = (joint_distributions[word_id][document_label] + 1) / (count_labels[document_label] + J)
    
        

    
    exit()

    doc = Document(path = 'test')
    print(doc.contents)
    nb = NaiveBayes()
    nb.classify(doc)
