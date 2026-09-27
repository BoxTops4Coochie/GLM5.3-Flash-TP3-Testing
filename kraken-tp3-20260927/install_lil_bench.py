"""Install the pinned llm-inference-bench as ``lil-bench`` into the runtime image.

``lil-bench`` runs inside a serving container (``docker exec … lil-bench``):
it measures the running server with the standard matrix and uploads the
result to docker.local-inference-lab.ai. The source is a pinned commit whose
archive checksum is locked; p2pmark is compiled for the image's GPUs against
the image's CUDA and NCCL.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

LOCK = Path(__file__).with_name("lil-bench.lock.json")
PREFIX = Path("/opt/lil/bench")
LAUNCHER = Path("/opt/venv/bin/lil-bench")
RECEIPT = Path("/opt/venv/share/lil-runtime/lil-bench.json")
# Blackwell data-center, RTX PRO/GeForce and DGX Spark, plus PTX for newer GPUs.
GENCODE = [
    "-gencode", "arch=compute_100,code=sm_100",
    "-gencode", "arch=compute_120,code=sm_120",
    "-gencode", "arch=compute_121,code=sm_121",
    "-gencode", "arch=compute_120,code=compute_120",
]


def archive_url(lock: dict) -> str:
    return f"https://codeload.github.com/{lock['repository']}/tar.gz/{lock['commit']}"


def fetch(lock: dict) -> bytes:
    request = urllib.request.Request(archive_url(lock), headers={"User-Agent": "lil-runtime-build"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def extract(data: bytes, lock: dict, prefix: Path) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != lock["archive_sha256"]:
        raise ValueError(f"llm-inference-bench archive checksum {digest} != locked {lock['archive_sha256']}")
    staging = prefix.with_name(prefix.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        top = members[0].name.split("/", 1)[0]
        for member in members:
            if not member.name.startswith(top + "/"):
                continue
            member.name = member.name[len(top) + 1:]
            if member.name:
                archive.extract(member, staging, filter="data")
    shutil.rmtree(prefix, ignore_errors=True)
    staging.rename(prefix)


def build_p2pmark(prefix: Path) -> Path:
    tool = prefix / "tools/p2pmark"
    binary = tool / "llm_p2pmark"
    binary.unlink(missing_ok=True)  # never ship a prebuilt binary from the repository
    subprocess.run(
        ["make", "-B", "-C", str(tool), "NVCC=/usr/local/cuda/bin/nvcc",
         "CFLAGS=-O2 -std=c++20 " + " ".join(GENCODE)],
        check=True,
    )
    if not os.access(binary, os.X_OK):
        raise ValueError("p2pmark did not build")
    return binary


def install_launcher(prefix: Path, launcher: Path) -> None:
    launcher.write_text(
        "#!/bin/sh\n"
        "# Standardized benchmark of this container: docker exec -it -e LIL_BENCH_TOKEN=... <container> lil-bench\n"
        f'PYTHONPATH="{prefix}${{PYTHONPATH:+:$PYTHONPATH}}" exec /opt/venv/bin/python -m lil_bench "$@"\n'
    )
    launcher.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    extract(fetch(lock), lock, PREFIX)
    binary = build_p2pmark(PREFIX)
    install_launcher(PREFIX, LAUNCHER)
    subprocess.run([str(LAUNCHER), "--help"], check=True, stdout=subprocess.DEVNULL)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps({
        **lock,
        "prefix": str(PREFIX),
        "launcher": str(LAUNCHER),
        "p2pmark_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
