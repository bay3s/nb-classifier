from src.NaiveBayes import NaiveBayes
import pathlib


if __name__ == '__main__':
    parent_path = pathlib.Path(__file__).parent.resolve()
    nb = NaiveBayes(
        vocab_path = f'{parent_path}/data/vocabulary.txt',
        labels_path = f'{parent_path}/data/newsgrouplabels.txt',
        train_data_path = f'{parent_path}/data/train.data',
        train_labels_path = f'{parent_path}/data/train.label',
        test_data_path = f'{parent_path}/data/test.data',
        test_labels_path = f'{parent_path}/data/test.label'
    )
    
    nb.train()
    predictions = nb.test()

    exit()
