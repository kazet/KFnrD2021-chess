import os
from pathlib import Path
from typing import List

import dotenv
import typer

from downloader import download_data
from preprocessor import preprocess_data


def main(datasets: List[str] = typer.Argument(..., help='Names of datasets in format `yyyy-mm`'),
         env: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                  resolve_path=True, help='Path to file with configuration enviromental variables'
                                  )) -> None:
    """
    Download datasets from lichess.org and perform needed preprocessing.
    Files will be saved into directory passed in variable `DATASETS` in given .env file.
    """
    # both env and datasets are required, env is an option so it's typed with `--env`
    datasets_dir = dotenv.dotenv_values(env)['DATASETS']
    if not os.path.isdir(datasets_dir):
        raise FileNotFoundError('Folder given to save datasets does not exist.')
    for month in set(datasets):  # use datasets given by user
        filename = download_data(month)
        out_path = os.path.join(datasets_dir, filename)
        preprocess_data(filename, out_path)
    typer.echo(f'Successfully downloaded and preproccessed {len(set(datasets))} file(s).')


if __name__ == '__main__':
    typer.run(main)
