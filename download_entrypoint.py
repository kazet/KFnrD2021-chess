import sys

from downloader import download_data
from preprocessor import preprocess_data


DATASETS = []  # list of needed datasets in form `yyyy-mm`


def main():
    paths = []  # locations of downloaded datasets
    for month in set(DATASETS + sys.argv[1:]):  # use also datasets given by user
        path = download_data(month)
        path = preprocess_data(path)
        paths.append(path)


if __name__ == '__main__':
    main()
