from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "kaggle"

DEFAULT_DATASETS = {
    "savee": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
    "animated": "ziya07/multimodal-dataset-for-animated-content-analysis",
}


def kaggle_credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def copy_downloaded_dataset(downloaded_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if downloaded_path.is_file():
        shutil.copy2(downloaded_path, target_dir / downloaded_path.name)
        return
    for item in downloaded_path.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def run_kagglehub_download(slug: str, output_dir: Path) -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub first: pip install kagglehub[pandas-datasets]") from exc

    downloaded_path = Path(kagglehub.dataset_download(slug))
    target_dir = output_dir / slug.replace("/", "__")
    copy_downloaded_dataset(downloaded_path, target_dir)
    print(f"Copied {slug} from KaggleHub cache to {target_dir}")


def run_kaggle_cli_download(slug: str, output_dir: Path, unzip: bool) -> None:
    target_dir = output_dir / slug.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(target_dir),
    ]
    if unzip:
        command.append("--unzip")
    subprocess.run(command, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download external Kaggle datasets for emotion recognition.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Where downloaded datasets should be stored")
    parser.add_argument("--dataset", action="append", choices=list(DEFAULT_DATASETS), help="Named dataset to download")
    parser.add_argument("--slug", action="append", default=[], help="Extra Kaggle dataset slug, for example owner/dataset-name")
    parser.add_argument("--method", choices=["kagglehub", "kaggle-cli"], default="kagglehub", help="Download backend")
    parser.add_argument("--no-unzip", action="store_true", help="Keep downloaded zip files compressed")
    parser.add_argument("--print-config", action="store_true", help="Print configured dataset slugs and exit")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_names = args.dataset or list(DEFAULT_DATASETS)
    slugs = [DEFAULT_DATASETS[name] for name in selected_names] + args.slug

    if args.print_config:
        print(json.dumps({"output_dir": str(args.output_dir), "method": args.method, "datasets": slugs}, indent=2))
        return

    if args.method == "kaggle-cli" and not kaggle_credentials_available():
        raise RuntimeError(
            "Kaggle credentials were not found. Create an API token from Kaggle, then place it at "
            f"{Path.home() / '.kaggle' / 'kaggle.json'} or set KAGGLE_USERNAME and KAGGLE_KEY."
        )

    for slug in slugs:
        print(f"Downloading {slug} ...")
        if args.method == "kagglehub":
            run_kagglehub_download(slug, args.output_dir)
        else:
            run_kaggle_cli_download(slug, args.output_dir, unzip=not args.no_unzip)
    print(f"Downloaded {len(slugs)} dataset(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
