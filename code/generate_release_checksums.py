"""Create or verify SHA-256 checksums for the public release files."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CHECKSUMS.sha256"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yml", ".yaml"}
TEXT_FILENAMES = {".gitattributes", ".gitignore"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def included_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not EXCLUDED_PARTS.intersection(path.parts)
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
        # Match the LF line endings enforced by .gitattributes, even when this
        # script runs from a Windows working tree containing CRLF files.
        value.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return value.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rendered_lines() -> list[str]:
    return [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in included_files()
    ]


def verify_manifest() -> int:
    lines = OUTPUT.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("[FAIL] Release checksum manifest is empty")
    failures: list[str] = []
    seen: set[str] = set()
    root = ROOT.resolve()
    for line_number, line in enumerate(lines, start=1):
        checksum, separator, relative = line.partition("  ")
        if not separator or not SHA256_PATTERN.fullmatch(checksum) or not relative:
            failures.append(f"line {line_number}: invalid manifest entry")
            continue
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(root) or relative in seen:
            failures.append(f"line {line_number}: unsafe or duplicate path: {relative}")
            continue
        seen.add(relative)
        if not path.is_file():
            failures.append(f"missing file: {relative}")
        elif digest(path) != checksum:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        details = "\n".join(f" - {failure}" for failure in failures)
        raise SystemExit(f"[FAIL] Release checksum verification failed\n{details}")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if not OUTPUT.is_file():
            raise FileNotFoundError(OUTPUT)
        count = verify_manifest()
        print(f"[PASS] Verified {count} release-file checksums")
        return
    expected = "\n".join(rendered_lines()) + "\n"
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"[DONE] Wrote {len(included_files())} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
