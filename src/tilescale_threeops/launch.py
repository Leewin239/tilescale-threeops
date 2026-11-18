"""Linux training launcher with timeout cleanup across torchrun sessions."""

import argparse
import os
from pathlib import Path
import signal
import subprocess


def descendants(root):
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
    found = {root}
    while True:
        extra = {pid for pid, parent in parents.items() if parent in found} - found
        if not extra:
            return found
        found.update(extra)


def send(pids, sig):
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--grace', type=float, default=20)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.timeout <= 0 or args.grace < 0 or not command:
        parser.error('Expected positive timeout, nonnegative grace and a command')
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        workers = descendants(process.pid)
        send(workers, signal.SIGINT)
        try:
            process.wait(timeout=args.grace)
        except subprocess.TimeoutExpired:
            pass
        # torchrun workers can create separate sessions and outlive their launcher.
        workers.update(descendants(process.pid))
        send(workers, signal.SIGKILL)
        process.wait()
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
