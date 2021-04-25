import hashlib
import os
import re

import requests


DATASETS_LIST_URL = 'https://database.lichess.org/standard/list.txt'  # list of urls to all files with datasets
CHECKSUMS_LIST_URL = 'https://database.lichess.org/standard/sha256sums.txt'  # list of sha256 checksums of data files


def download_data(month: str) -> str:
    """Function to download given dataset from lichess and save it

    :param month: month of dataset among available in lichess.org in form `yyyy-mm`

    :return: name of file containing downloaded data
    """
    if re.match(r'^\d{4}-(0[1-9]|1[0-2])$', month) is None:  # check for proper format
        raise ValueError('Month parameter should be in form `yyyy-mm`')
    datasets_list = '\n' + requests.get(DATASETS_LIST_URL).text + '\n'
    # get list of datasets with newline at beginning and end of file
    match = re.search(f'\n.+{month}.+\n', datasets_list)  # find url for given month, each is in new line
    if match is None:
        raise ValueError(f'Dataset from {month} cannot be found on lichess.org')
    url = match.group().strip()
    sha256 = hashlib.sha256()
    filename = url.rpartition('/')[-1]  # extract filename from url
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                sha256.update(chunk)
    checksums = requests.get(CHECKSUMS_LIST_URL).text
    checksum = sha256.hexdigest()
    if checksum + '  ' + filename not in checksums:  # checksums is list of sha256 and filenames separated by 2 spaces
        print('File downloaded inproperly. Retrying...')
        os.remove(filename)
        return download_data(month)
    return filename
