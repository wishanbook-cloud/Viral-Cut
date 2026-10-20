"""Smoke-test the packaged AI executable and its bundled runtime imports.

Run this from the AI build environment after ``build_windows.ps1 -IncludeAI``:

    ./.venv-ai/Scripts/python.exe ./packaging_smoke_test.py

The one-file executable normally extracts to a private temporary directory,
which makes importing its modules from a second process awkward. This check
uses PyInstaller's read-only archive readers to extract the package payload to
a temporary directory, then imports the same modules without site-packages on
``sys.path``. It also runs the executable's existing ``--smoke-test`` path.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader


ROOT = Path(__file__).resolve().parent
DEFAULT_EXE = ROOT / "dist-pro-ai" / "VideoEditorToolProAI.exe"


class _ZlibModuleLoader(importlib.abc.Loader):
    def __init__(
        self,
        archive: ZlibArchiveReader,
        name: str,
        origin: Path,
        package_path: Path | None,
    ) -> None:
        self.archive = archive
        self.name = name
        self.origin = origin
        self.package_path = package_path

    def create_module(self, spec: Any) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self.origin)
        if self.package_path is not None:
            module.__path__ = [str(self.package_path)]
        code = self.archive.extract(self.name)
        exec(code, module.__dict__)


class _ZlibModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, archive: ZlibArchiveReader, root: Path) -> None:
        self.archive = archive
        self.root = root

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: ModuleType | None = None,
    ) -> Any:
        entry = self.archive.toc.get(fullname)
        if entry is None:
            return None
        is_package = bool(entry[0])
        package_path = self.root.joinpath(*fullname.split("."))
        filename = "__init__.py" if is_package else f"{fullname.rsplit('.', 1)[-1]}.py"
        origin = package_path / filename if is_package else package_path.parent / filename
        loader = _ZlibModuleLoader(
            self.archive,
            fullname,
            origin,
            package_path if is_package else None,
        )
        return importlib.util.spec_from_loader(fullname, loader, is_package=is_package)


def _extract_onefile(executable: Path, destination: Path) -> ZlibArchiveReader:
    archive = CArchiveReader(str(executable))
    pyz_path = destination / "PYZ.pyz"
    pyz_path.write_bytes(archive.extract("PYZ.pyz"))

    # Native extensions, package source files and model/data assets are stored
    # in the outer CArchive. Root-level bootstrapping entries are not needed by
    # the import check and are intentionally left out.
    for name in archive.toc:
        if "\\" not in name or name == "PYZ.pyz":
            continue
        output = destination.joinpath(*name.replace("\\", "/").split("/"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(archive.extract(name))

    return ZlibArchiveReader(str(pyz_path))


def _check_bundled_imports(executable: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix="video-editor-ai-smoke-", ignore_cleanup_errors=True
    ) as temporary:
        extracted = Path(temporary)
        pyz = _extract_onefile(executable, extracted)
        finder = _ZlibModuleFinder(pyz, extracted)
        original_path = sys.path[:]
        original_meta_path = sys.meta_path[:]
        try:
            # Do not let imports silently fall back to the AI virtualenv. The
            # standard library remains available, while bundled packages must
            # come from the extracted executable payload.
            sys.path[:] = [str(extracted)] + [
                item for item in original_path if "site-packages" not in item.lower()
            ]
            sys.meta_path.insert(0, finder)
            cv2 = importlib.import_module("cv2")
            numpy = importlib.import_module("numpy")
            faster_whisper = importlib.import_module("faster_whisper")
            tqdm = importlib.import_module("tqdm")
            ctranslate2 = importlib.import_module("ctranslate2")
            assert hasattr(cv2, "CascadeClassifier")
            assert hasattr(numpy, "array")
            assert hasattr(faster_whisper, "WhisperModel")
            assert callable(tqdm.tqdm)
            assert hasattr(ctranslate2, "get_cuda_device_count")
            print(
                "Packaged imports passed: cv2, numpy, faster_whisper, tqdm, ctranslate2"
            )
        finally:
            sys.path[:] = original_path
            sys.meta_path[:] = original_meta_path


def main() -> int:
    executable = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_EXE
    if not executable.is_file():
        raise FileNotFoundError(f"Packaged AI executable was not found: {executable}")
    for sibling in ("ffmpeg.exe", "ffprobe.exe"):
        path = executable.parent / sibling
        if not path.is_file():
            raise FileNotFoundError(f"Packaged FFmpeg sibling was not found: {path}")

    result = subprocess.run([str(executable), "--smoke-test"], check=False)
    if result.returncode:
        raise RuntimeError(f"Packaged UI smoke test failed with exit code {result.returncode}")
    print(f"Packaged UI smoke test passed: {executable}")
    _check_bundled_imports(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
