import torch
import typer
from pathlib import Path

import settings
import model
from inference import INF


def main(
    path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to file with fens",
    )
):
    test = INF(
        torch.device("cpu"),
        model.Coder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE),
    )
    with open(path) as f:
        fens = f.read().split("\n")
    print("Results : \n", test.predict(fens))


if __name__ == "__main__":
    typer.run(main)
