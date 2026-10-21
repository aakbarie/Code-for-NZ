import hashlib

from morrill import raw

HEADER = "local_path,drive_id,drive_path,size_bytes,sha256,role,note\n"


def write_manifest(path, rows):
    path.write_text(HEADER + "".join(",".join(map(str, r)) + "\n" for r in rows))


def test_verify_reports_each_failure_mode(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    (raw_dir / "a").mkdir(parents=True)
    good = b"x,y\n1,2\n"
    (raw_dir / "a" / "good.csv").write_bytes(good)
    (raw_dir / "a" / "short.csv").write_bytes(good[:3])
    (raw_dir / "a" / "edited.csv").write_bytes(b"x,y\n1,3\n")
    digest = hashlib.sha256(good).hexdigest()
    manifest = tmp_path / "raw_files.csv"
    write_manifest(
        manifest,
        [
            (f"a/{name}", "ID", "My Drive/x", len(good), digest, "test", "")
            for name in ("good.csv", "short.csv", "edited.csv", "missing.csv")
        ],
    )

    assert raw.verify(raw_dir, manifest) is False
    out = capsys.readouterr().out
    assert "ok  a/good.csv" in out
    assert "wrong size  a/short.csv" in out
    assert "wrong checksum  a/edited.csv" in out
    assert "missing  a/missing.csv" in out
    assert "https://drive.google.com/file/d/ID/view" in out
    assert "1/4 files OK" in out


def test_verify_passes_when_everything_matches(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "f.txt").write_bytes(b"hello")
    manifest = tmp_path / "raw_files.csv"
    write_manifest(
        manifest,
        [("f.txt", "ID", "p", 5, hashlib.sha256(b"hello").hexdigest(), "test", "")],
    )
    assert raw.verify(raw_dir, manifest) is True


def test_entry_without_checksum_is_checked_by_size(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "big.csv").write_bytes(b"12345")
    manifest = tmp_path / "raw_files.csv"
    write_manifest(manifest, [("big.csv", "ID", "p", 5, "", "test", "")])
    assert raw.verify(raw_dir, manifest) is True
    assert "ok (size only)  big.csv" in capsys.readouterr().out
