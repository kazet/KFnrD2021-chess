import torch
import typer
from pathlib import Path

import settings
from model import Coder
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
    ),
):
    model = Coder(settings.BOARD_SHAPE, settings.LATENT_SIZE).to(settings.DEVICE)
    model.load_state_dict(torch.load(setting.CODER_PATH))
    model.eval()

    test = Inference(
        settings.DEVICE,
        model,
    )
    with open(path) as f:
        fens = f.read().split("\n")
    print("Results : \n", test.predict(fens))


if __name__ == "__main__":
    typer.run(main)
