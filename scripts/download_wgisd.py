"""Descarga el dataset WGISD (Embrapa Wine Grape Instance Segmentation Dataset).

Fuente: repositorio GitHub thsant/wgisd. Es público, no requiere cuenta.
Licencia CC BY-NC 4.0 - uso académico, citar Santos et al. 2020 (ver README).
"""
import argparse
import subprocess
from pathlib import Path

WGISD_REPO_URL = "https://github.com/thsant/wgisd.git"


def download(dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} ya existe y no está vacío, no se vuelve a clonar.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", WGISD_REPO_URL, str(dest)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga WGISD por git clone")
    parser.add_argument("--dest", default="data/wgisd", help="carpeta destino")
    args = parser.parse_args()
    download(Path(args.dest))
