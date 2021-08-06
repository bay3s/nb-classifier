from src.NaiveBayes import NaiveBayes
import pathlib


if __name__ == '__main__':
    parent_path = pathlib.Path(__file__).parent.resolve()
    nb = NaiveBayes(
        num_categories = 20,
        vocab_file_location = f'{parent_path}/data/vocabulary.txt',
        training_data_location = f'{parent_path}/data/train.data',
        training_labels_location = f'{parent_path}/data/train.label',
        test_data_location = f'{parent_path}/data/test.data',
        test_labels_location = f'{parent_path}/data/test.label'
    )

    nb.train()
    results = nb.run_classifier()
    exit()
