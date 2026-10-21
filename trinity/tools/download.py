# trinity/tools/download.py
"""
Download and extract Trinity resources and datasets (or any custom ZIP).

Usage
-----
# download both resource and dataset
python -m trinity.tools.download

# resource only
python -m trinity.tools.download --resource

# dataset only
python -m trinity.tools.download --dataset

# custom ZIP for debugging
python -m trinity.tools.download --url https://example.com/test.zip --dir ./sandbox
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = "Trinity-UST/Trinity"
VERSION = "v1.0.0"

DEFAULT_ASSETS = {
    "resource": f"https://github.com/{REPO}/releases/download/{VERSION}/trinity_resource.zip",
    "dataset":  f"https://github.com/{REPO}/releases/download/{VERSION}/trinity_dataset.zip",
}

def download(url: str, dst: Path) -> None:
    """Stream-download *url* to *dst* with a progress indicator."""
    print(f"▶ downloading {url.split('/')[-1]} …")
    with urllib.request.urlopen(url) as response, open(dst, "wb") as f:
        length = response.length or 0
        block = 32 * 1024
        done = 0
        while True:
            chunk = response.read(block)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if length:
                filled = int(50 * done / length)
                bar = "█" * filled + "." * (50 - filled)
                mb_now = done / 1_048_576
                mb_tot = length / 1_048_576
                sys.stdout.write(f"\r    [{bar}] {mb_now:.1f}/{mb_tot:.1f} MB")
                sys.stdout.flush()
    print("\n    download complete")


def extract(zip_path: Path, dest_dir: Path) -> None:
    """Unzip *zip_path* into *dest_dir* and delete the archive."""
    print(f"▶ extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path=dest_dir)
    zip_path.unlink()


def install(assets: dict[str, str], trinity_dir: Path) -> None:
    """Download and unpack selected assets into *trinity_dir*."""
    trinity_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for url in assets.values():
            zip_path = tmp / url.split("/")[-1]
            download(url, zip_path)
            extract(zip_path, trinity_dir)

    print("\nDone!")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download Trinity assets.")
    ap.add_argument("--resource", action="store_true", help="download trinity_resource.zip")
    ap.add_argument("--dataset", action="store_true", help="download trinity_dataset.zip")
    ap.add_argument("--url", help="download a custom ZIP (debug/testing)")
    ap.add_argument(
        "--dir",
        default=os.getenv("TRINITY_DIR", "."),
        help="installation directory (default: $TRINITY_DIR or current dir)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.url:  # custom single file
        assets = {"custom": args.url}
    else:
        if not args.resource and not args.dataset:  # default: both
            assets = DEFAULT_ASSETS
        else:  # selective
            assets = {
                k: v
                for k, v in DEFAULT_ASSETS.items()
                if (k == "resource" and args.resource) or (k == "dataset" and args.dataset)
            }

    install(assets, Path(args.dir).expanduser().resolve())


if __name__ == "__main__":
    main()