#!/usr/bin/env python3
"""Other processes' use of one GPU, from DRM fdinfo (i915 and xe), for guard.sh.

  drm_clients.py --pdev 0000:03:00.0 [--interval 1.0] [--proc /proc]

Samples every readable /proc/PID/fdinfo entry of the card twice, INTERVAL seconds apart, and prints

  max <percent> <engine> <pid> <comm>          the busiest client and engine (max 0.0 - - - when none)
  <percent> <engine> <pid> <comm>              one line per client and engine above 0.1%

i915 reports busy nanoseconds per engine class (drm-engine-render: N ns, divided by drm-engine-capacity-*
when a class has several engines); xe reports busy and total GPU cycles (drm-cycles-rcs / drm-total-cycles-rcs).
Processes of other users are only visible as root, so run guard.sh as root to see a display server.
"""

import argparse
import glob
import os
import re
import time

LINE = re.compile(r"^(drm-[a-z0-9-]+):\s+(\S+)")


def read_clients(proc, pdev):
    """{(client id): (pid, comm, {key: value})} for fds of this card; one entry per DRM client."""
    clients = {}
    for fd in glob.glob(os.path.join(proc, "[0-9]*", "fdinfo", "[0-9]*")):
        try:
            with open(fd) as f:
                text = f.read()
        except OSError:
            continue
        if "drm-client-id" not in text:
            continue
        kv = {}
        for line in text.splitlines():
            m = LINE.match(line)
            if m:
                kv[m.group(1)] = m.group(2)
        if kv.get("drm-pdev") != pdev or "drm-client-id" not in kv:
            continue
        cid = kv["drm-client-id"]
        if cid in clients:
            continue
        pid = fd.split(os.sep)[-3]
        try:
            with open(os.path.join(proc, pid, "comm")) as f:
                comm = f.read().strip()
        except OSError:
            comm = "?"
        clients[cid] = (pid, comm, kv)
    return clients


def busy(before, after, wall_ns):
    """[(percent, engine)] for one client between two samples."""
    out = []
    for key, val in after.items():
        if key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity-"):  # i915, "N ns"
            eng = key[len("drm-engine-"):]
            try:
                d = int(val) - int(before.get(key, val))
            except ValueError:
                continue
            cap = int(after.get("drm-engine-capacity-" + eng, "1"))
            out.append((100.0 * d / (wall_ns * cap), eng))
        elif key.startswith("drm-cycles-"):  # xe, busy cycles against the GPU's total cycles
            eng = key[len("drm-cycles-"):]
            tkey = "drm-total-cycles-" + eng
            try:
                d = int(val) - int(before.get(key, val))
                t = int(after.get(tkey, "0")) - int(before.get(tkey, after.get(tkey, "0")))
            except ValueError:
                continue
            if t > 0:
                out.append((100.0 * d / t, eng))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdev", required=True, help="PCI address of the card, as drm-pdev shows it")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--proc", default="/proc")
    a = ap.parse_args()
    t0 = time.monotonic_ns()
    first = read_clients(a.proc, a.pdev)
    time.sleep(a.interval)
    second = read_clients(a.proc, a.pdev)
    wall = time.monotonic_ns() - t0
    rows = []
    for cid, (pid, comm, kv) in second.items():
        if cid not in first or first[cid][0] != pid:
            continue
        for pct, eng in busy(first[cid][2], kv, wall):
            rows.append((pct, eng, pid, comm))
    rows.sort(reverse=True)
    top = rows[0] if rows else (0.0, "-", "-", "-")
    print(f"max {top[0]:.1f} {top[1]} {top[2]} {top[3]}")
    for pct, eng, pid, comm in rows:
        if pct > 0.1:
            print(f"{pct:.1f} {eng} {pid} {comm}")


if __name__ == "__main__":
    main()
