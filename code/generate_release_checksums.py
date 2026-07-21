"""Create or verify SHA-256 checksums for the public release files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CHECKSUMS.sha256"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}


def included_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not EXCLUDED_PARTS.intersection(path.parts)
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rendered_lines() -> list[str]:
    return [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in included_files()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = "\n".join(rendered_lines()) + "\n"
    if args.verify:
        if not OUTPUT.is_file():
            raise FileNotFoundError(OUTPUT)
        if OUTPUT.read_text(encoding="utf-8") != expected:
            raise SystemExit("[FAIL] CHECKSUMS.sha256 is out of date")
        print(f"[PASS] Verified {len(included_files())} release-file checksums")
        return
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"[DONE] Wrote {len(included_files())} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
