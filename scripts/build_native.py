"""Build the small DLPack descriptor helper using the active PyTorch headers."""

import argparse
import subprocess
from pathlib import Path
from torch.utils.cpp_extension import include_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "src/tilescale_threeops/native"
    output = directory / "_vmm_dlpack.so"
    command = [
        args.cxx,
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        str(directory / "vmm_dlpack.cc"),
        "-o",
        str(output),
    ]
    command.extend("-I" + path for path in include_paths())
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
