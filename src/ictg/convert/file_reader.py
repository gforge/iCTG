import io
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import zipfile_deflate64  # noqa: F401 - monkey-patches zipfile

JSON_SUFFIX = ".json"
ZIP_SUFFIX = ".zip"


def _pick_zip_json_member(zf: zipfile.ZipFile) -> str:
    json_members = [zi for zi in zf.infolist() if zi.filename.lower().endswith(JSON_SUFFIX)]
    if not json_members:
        raise FileNotFoundError(f"No {JSON_SUFFIX} member found inside zip.")
    if len(json_members) == 1:
        return json_members[0].filename
    json_members.sort(key=lambda zi: zi.file_size, reverse=True)  # prefer the largest
    return json_members[0].filename


@contextmanager
def open_json_stream(
    path: Path, member: str | None = None
) -> Generator[io.TextIOWrapper, Any, None]:
    """
    Context manager that yields a readable binary stream for either a .json file or a .zip member.
    """
    if path.suffix.lower() == JSON_SUFFIX:
        with open(path, encoding="utf-8") as fp:
            yield fp
    elif path.suffix.lower() == ZIP_SUFFIX:
        with zipfile.ZipFile(path) as zf:
            name = member or _pick_zip_json_member(zf)
            with zf.open(name, mode="r") as fp:
                with io.TextIOWrapper(fp, encoding="utf-8") as text_fp:
                    yield text_fp
    else:
        raise ValueError(f"Unsupported input: {path}")
