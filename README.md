# KFnrD2021-chess


### 1. Creating configuration
Make an `*.env` file for configuration data.
You have to put following variables:
* `DATASETS` - path to a directory to save preprocessed datasets

Syntax is `<key> = <value>\n`. Keys can be unquoted or single-quoted. 
Values can be unquoted, single- or double-quoted.


### 2. Downloading data
Use `python3 download_entrypoint.py --env <env_file> <datasets..>` 
to download datasets and perform the necessary preprocessing.
You have to replace `<env-file>` with path to file with configuration data 
and `<datasets...>` with names of datasets
from [lichess.org](http://database.lichess.org) in format `yyyy-mm`.

Use `python3 download_entrypoint.py --help` for help.