"""Check the public reproduction environment.

Run this after installing dependencies:

    python code/check_environment.py

The script verifies imports, prints package versions, and reports whether CUDA
is visible to PyTorch. CUDA availability is not required for correctness, but
full 6 x 17 x 5 training is expected to be slow on CPU.
"""

from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_PACKAGES = {
    "torch": "2.9",
    "torchvision": "0.24",
    "timm": "1.0",
    "albumentations": "1.4",
    "cv2": "4.12",
    "PIL": "12.",
    "numpy": "2.2",
    "pandas": "3.0",
    "sklearn": "1.7",
    "scipy": "1.14",
    "statsmodels": "0.14",
    "matplotlib": "3.9",
    "seaborn": "0.13",
    "tqdm": "4.67",
    "openpyxl": "3.1",
}


def package_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    if module_name == "PIL":
        import PIL

        return PIL.__version__
    if module_name == "sklearn":
        import sklearn

        return sklearn.__version__
    return getattr(module, "__version__", "unknown")


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print()

    failures = []
    for package, expected_prefix in REQUIRED_PACKAGES.items():
        try:
            version = package_version(package)
            status = "OK" if version.startswith(expected_prefix) else "CHECK"
            print(f"{status:5s} {package:15s} {version}")
        except Exception as exc:
            failures.append((package, str(exc)))
            print(f"MISS  {package:15s} {exc}")

    print()
    try:
        import torch

        print("Torch CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device 0:", torch.cuda.get_device_name(0))
            print("Torch CUDA build:", torch.version.cuda)
        else:
            print("CUDA note: training will run on CPU unless a CUDA-enabled PyTorch build is installed.")
    except Exception as exc:
        failures.append(("torch_cuda_check", str(exc)))

    if failures:
        print()
        print("Environment check failed for:")
        for package, reason in failures:
            print(f"- {package}: {reason}")
        raise SystemExit(1)

    print()
    print("[OK] Environment import check completed.")


if __name__ == "__main__":
    main()
