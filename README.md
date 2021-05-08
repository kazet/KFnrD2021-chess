# KFnrD2021-chess


### 1. Creating configuration
Make an `.env` file for configuration data in a main directory of the repository.
You have to put following variables:
* `DATASETS` - path to a directory to save preprocessed datasets, directory has to exist.

Syntax is `<key> = <value>\n`. Keys can be unquoted or single-quoted. 
Values can be unquoted, single- or double-quoted.


### 2. Downloading data
Use `python3 download_entrypoint.py <datasets..>` to download 
datasets and perform the necessary preprocessing.
You have to replace `<datasets...>` with names of datasets
from [lichess.org](https://database.lichess.org) in format 
`yyyy-mm`.

![Scheme](https://i.ibb.co/6gNJ6XZ/scheme2.jpg)
