from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

DEFAULT_RECORD_ID = "1188976"
DEFAULT_DATA_ROOT = Path("E:/emotion_recognition_data")
DEFAULT_DOWNLOAD_DIR = DEFAULT_DATA_ROOT / "downloads" / "ravdess_zenodo"
DEFAULT_EXTRACT_DIR = DEFAULT_DATA_ROOT / "raw" / "ravdess_full"


def fetch_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urlopen(Request(url, headers={"User-Agent": "emotion-recognition-internship/1.0"}), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def selected_actor_ids(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return {f"{index:02d}" for index in range(1, 25)}
    actors: set[str] = set()
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"Actor id must be numeric, got {part!r}")
        actor_id = int(part)
        if actor_id < 1 or actor_id > 24:
            raise ValueError(f"Actor id must be between 1 and 24, got {actor_id}")
        actors.add(f"{actor_id:02d}")
    if not actors:
        raise ValueError("No actors selected.")
    return actors


def find_actor_files(record: dict, actors: set[str], include_song: bool) -> list[dict]:
    files = record.get("files") or []
    selected: list[dict] = []
    speech_pattern = re.compile(r"^Video_Speech_Actor_(\d{2})\.zip$")
    song_pattern = re.compile(r"^Video_Song_Actor_(\d{2})\.zip$")
    for item in files:
        key = str(item.get("key") or "")
        speech_match = speech_pattern.match(key)
        song_match = song_pattern.match(key)
        actor = speech_match.group(1) if speech_match else song_match.group(1) if song_match else None
        if actor is None or actor not in actors:
            continue
        if song_match and not include_song:
            continue
        selected.append(item)
    selected.sort(key=lambda item: str(item.get("key") or ""))
    missing = sorted(actors - {re.search(r"Actor_(\d{2})", str(item.get("key") or "")).group(1) for item in selected})
    if missing:
        raise RuntimeError(f"Zenodo record did not contain selected actor zip(s): {missing}")
    return selected


def file_size(item: dict) -> int | None:
    size = item.get("size")
    if isinstance(size, int):
        return size
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def download_url(item: dict) -> str:
    links = item.get("links") or {}
    url = links.get("self") or links.get("download")
    if not url:
        raise RuntimeError(f"No download link found for {item.get('key')}")
    return str(url)


def download_file(item: dict, download_dir: Path, overwrite: bool) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    name = str(item["key"])
    expected_size = file_size(item)
    output_path = download_dir / name
    part_path = download_dir / f"{name}.part"

    if output_path.exists() and not overwrite:
        if expected_size is None or output_path.stat().st_size == expected_size:
            print(f"Already downloaded: {output_path}")
            return output_path
        print(f"Existing file has unexpected size; redownloading: {output_path}")
        output_path.unlink()

    if overwrite and output_path.exists():
        output_path.unlink()
    if overwrite and part_path.exists():
        part_path.unlink()

    resume_from = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "emotion-recognition-internship/1.0"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    request = Request(download_url(item), headers=headers)
    try:
        response = urlopen(request, timeout=120)
    except HTTPError as exc:
        if resume_from and exc.code == 416:
            part_path.rename(output_path)
            return output_path
        raise
    except URLError:
        raise

    if resume_from and response.status != 206:
        resume_from = 0
        part_path.unlink(missing_ok=True)

    total = expected_size or int(response.headers.get("Content-Length") or 0) + resume_from
    mode = "ab" if resume_from else "wb"
    with response, part_path.open(mode) as handle, tqdm(
        total=total,
        initial=resume_from,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=name,
    ) as progress:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            progress.update(len(chunk))

    if expected_size is not None and part_path.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch for {name}: got {part_path.stat().st_size}, expected {expected_size}")
    part_path.replace(output_path)
    return output_path


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    marker = extract_dir / ".extracted" / zip_path.stem
    if marker.exists() and not overwrite:
        print(f"Already extracted: {zip_path.name}")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        if overwrite:
            for member in archive.namelist():
                target = extract_dir / member
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
        archive.extractall(extract_dir)
    marker.write_text(zip_path.name, encoding="utf-8")
    print(f"Extracted: {zip_path.name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and extract full RAVDESS actor video-speech zips from Zenodo.")
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--actors", default="all", help="Comma-separated actors, e.g. 01,02,03, or 'all'.")
    parser.add_argument("--include-song", action="store_true", help="Also download Video_Song actor zips. Not recommended for the 7-class speech model.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    actors = selected_actor_ids(args.actors)
    if args.extract_only:
        zips = sorted(args.download_dir.glob("Video_Speech_Actor_*.zip"))
        if args.include_song:
            zips += sorted(args.download_dir.glob("Video_Song_Actor_*.zip"))
    else:
        record = fetch_record(args.record_id)
        zips = [
            download_file(item, args.download_dir, overwrite=args.overwrite)
            for item in find_actor_files(record, actors, include_song=args.include_song)
        ]

    if not args.download_only:
        for zip_path in zips:
            extract_zip(zip_path, args.extract_dir, overwrite=args.overwrite)

    print(f"Ready zips: {len(zips)}")
    print(f"Download directory: {args.download_dir}")
    print(f"Extract directory: {args.extract_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
