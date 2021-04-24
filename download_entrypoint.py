import os
import sys

from downloader import download_data
from preprocessor import preprocess_data
from settings import DATASETS


def main():
    if not os.path.isdir(DATASETS):
        raise FileNotFoundError('Folder given to save datasets does not exist.')
    datasets = sys.argv[1:]  # use datasets given by user
    for month in set(datasets):
        filename = download_data(month)
        out_path = os.path.join(DATASETS, filename)
        preprocess_data(filename, out_path)
    print(f'Successfully downloaded and preproccessed {len(set(datasets))} file(s).')


if __name__ == '__main__':
    main()
