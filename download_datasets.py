from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


AMI_ANNOTATIONS_VERSION = "1.6.2"
AMI_ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)
AMI_AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    "{meeting}/audio/{meeting}.Mix-Headset.wav"
)
AMI_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

QMSUM_COMMIT = "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e"
QMSUM_URL = f"https://github.com/Yale-LILY/QMSum/archive/{QMSUM_COMMIT}.zip"
QMSUM_LICENSE_URL = "https://github.com/Yale-LILY/QMSum/blob/main/LICENSE"

DEFAULT_AMI_MEETINGS = ("ES2008a", "ES2008b", "ES2008c", "ES2008d")
USER_AGENT = "meeting-ai-dataset-downloader/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Download URL to destination without exposing partial files as complete."""
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Reuse: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    print(f"Download: {url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open(
            "wb"
        ) as output:
            expected = int(response.headers.get("Content-Length", "0"))
            received = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if expected:
                    percent = received * 100 / expected
                    print(
                        f"  {received / 1024**2:7.1f} MiB / "
                        f"{expected / 1024**2:7.1f} MiB ({percent:5.1f}%)",
                        end="\r",
                        flush=True,
                    )
        if expected and received != expected:
            raise OSError(f"Expected {expected} bytes but received {received}")
        partial.replace(destination)
        print(f"Saved: {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def safe_member_path(member_name: str) -> PurePosixPath:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe ZIP member: {member_name}")
    return member


def extract_zip(archive: Path, destination: Path, *, strip_root: bool = False) -> None:
    marker = destination / ".extraction_complete"
    if marker.is_file():
        print(f"Reuse extracted data: {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt:
            raise zipfile.BadZipFile(f"CRC failure in {corrupt}")

        members = [info for info in bundle.infolist() if info.filename]
        roots = {
            safe_member_path(info.filename).parts[0]
            for info in members
            if safe_member_path(info.filename).parts
        }
        if strip_root and len(roots) != 1:
            raise ValueError(f"Expected one ZIP root directory, found: {sorted(roots)}")

        for info in members:
            member = safe_member_path(info.filename)
            parts = member.parts[1:] if strip_root else member.parts
            if not parts:
                continue
            target = destination.joinpath(*parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

    marker.write_text(
        f"archive={archive.name}\nsha256={sha256(archive)}\n",
        encoding="utf-8",
    )
    print(f"Extracted: {archive} -> {destination}")


def validate_wav(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as audio:
        details = {
            "channels": audio.getnchannels(),
            "sample_width_bytes": audio.getsampwidth(),
            "sample_rate_hz": audio.getframerate(),
            "frame_count": audio.getnframes(),
        }
    if details["frame_count"] <= 0 or details["sample_rate_hz"] <= 0:
        raise ValueError(f"Invalid WAV metadata: {path}")
    return details


def file_record(path: Path, *, source_url: str, license_name: str) -> dict[str, object]:
    return {
        "path": str(path),
        "source_url": source_url,
        "license": license_name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Download destination (default: data/raw inside the project)",
    )
    parser.add_argument(
        "--ami-meetings",
        nargs="*",
        default=list(DEFAULT_AMI_MEETINGS),
        help="AMI meeting IDs for headset-mix audio",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Download annotations and QMSum without AMI WAV files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    downloads_dir = data_dir / "downloads"
    records: list[dict[str, object]] = []

    ami_archive = downloads_dir / f"ami_public_manual_{AMI_ANNOTATIONS_VERSION}.zip"
    download(AMI_ANNOTATIONS_URL, ami_archive)
    extract_zip(ami_archive, data_dir / "ami" / "annotations")
    records.append(
        file_record(
            ami_archive,
            source_url=AMI_ANNOTATIONS_URL,
            license_name="CC BY 4.0",
        )
    )

    audio_details: dict[str, dict[str, int]] = {}
    if not args.skip_audio:
        for meeting in args.ami_meetings:
            if not meeting.replace("_", "").isalnum():
                raise ValueError(f"Invalid AMI meeting ID: {meeting!r}")
            url = AMI_AUDIO_URL.format(meeting=meeting)
            wav_path = data_dir / "ami" / "audio" / meeting / f"{meeting}.Mix-Headset.wav"
            download(url, wav_path)
            audio_details[meeting] = validate_wav(wav_path)
            records.append(
                file_record(wav_path, source_url=url, license_name="CC BY 4.0")
            )

    qmsum_archive = downloads_dir / f"qmsum-{QMSUM_COMMIT}.zip"
    download(QMSUM_URL, qmsum_archive)
    extract_zip(qmsum_archive, data_dir / "qmsum", strip_root=True)
    records.append(
        file_record(qmsum_archive, source_url=QMSUM_URL, license_name="MIT")
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "licenses": {
            "AMI": AMI_LICENSE_URL,
            "QMSum": QMSUM_LICENSE_URL,
        },
        "versions": {
            "AMI_manual_annotations": AMI_ANNOTATIONS_VERSION,
            "QMSum_commit": QMSUM_COMMIT,
        },
        "ami_audio_metadata": audio_details,
        "files": records,
    }
    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print("Dataset download completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

