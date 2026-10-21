"""Raw input files: the manifest of what we need, and a check that we have it.

Raw data stays in Google Drive. manifests/raw_files.csv lists every file the
database is built from, with its Drive file ID and SHA-256 checksum. Download
each file into data/raw/<local_path>, then run `morrill verify-raw` to confirm
nothing is missing, truncated, or a different version than the one the
database was designed around.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from morrill.paths import MANIFEST_PATH, RAW_DIR


@dataclass(frozen=True)
class RawFile:
    local_path: str
    drive_id: str
    drive_path: str
    size_bytes: int
    sha256: str
    role: str
    note: str

    @property
    def drive_url(self) -> str:
        return f"https://drive.google.com/file/d/{self.drive_id}/view"


def read_manifest(path: Path = MANIFEST_PATH) -> list[RawFile]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            RawFile(
                local_path=row["local_path"],
                drive_id=row["drive_id"],
                drive_path=row["drive_path"],
                size_bytes=int(row["size_bytes"]),
                sha256=row["sha256"],
                role=row["role"],
                note=row["note"],
            )
            for row in csv.DictReader(fh)
        ]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def check(entry: RawFile, raw_dir: Path = RAW_DIR) -> str:
    """Return 'ok', 'ok (size only)', 'missing', 'wrong size', or 'wrong checksum'.

    'ok (size only)': the manifest has no checksum yet for this file (it has
    never been downloaded where one could be computed), but the size matches.
    """
    path = raw_dir / entry.local_path
    if not path.exists():
        return "missing"
    if path.stat().st_size != entry.size_bytes:
        return "wrong size"
    if not entry.sha256:
        return "ok (size only)"
    if sha256_of(path) != entry.sha256:
        return "wrong checksum"
    return "ok"


def verify(raw_dir: Path = RAW_DIR, manifest: Path = MANIFEST_PATH) -> bool:
    entries = read_manifest(manifest)
    n_ok = 0
    for entry in entries:
        status = check(entry, raw_dir)
        n_ok += status.startswith("ok")
        print(f"{status:>15}  {entry.local_path}")
        if status != "ok":
            print(f"{'':>15}  get it from: {entry.drive_path}")
            print(f"{'':>15}               {entry.drive_url}")
    print(f"\n{n_ok}/{len(entries)} files OK")
    return n_ok == len(entries)
