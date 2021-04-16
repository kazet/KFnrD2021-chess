from downloader import download_data
from preprocessor import preprocess_data


DATASETS = ['2013-01']  # list of needed datasets in form `yyyy-mm` # TODO: Add needed datasets


def main():
    paths = []  # locations of downloaded datasets
    for month in DATASETS:
        path = download_data(month)
        path = preprocess_data(path)
        paths.append(path)


if __name__ == '__main__':
    main()
