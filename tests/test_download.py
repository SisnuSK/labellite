import hashlib

import pytest

from labellite.data.download import fetch, safe_extract
from labellite.data.sources import RemoteFile


def _remote(src, md5):
    return RemoteFile(src.as_uri(), "out.bin", src.stat().st_size, md5)


def test_fetch_verifies_md5(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"labellite" * 100)
    good = hashlib.md5(src.read_bytes()).hexdigest()
    path = fetch(_remote(src, good), tmp_path / "dl")
    assert path.read_bytes() == src.read_bytes()
    with pytest.raises(RuntimeError, match="MD5"):
        fetch(_remote(src, "0" * 32), tmp_path / "dl2")


def test_safe_extract_rejects_path_traversal(tmp_path):
    import zipfile

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError, match="unsafe"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
