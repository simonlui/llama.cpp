#!/usr/bin/env python3
"""Register, spill and DPAS check for one ggml-sycl source file, compiled ahead of time for one device.

  isa_check.py BUILD_DIR FILE [--device acm-g10] [--out DIR] [--filter REGEX] [-D NAME[=VALUE] ...]
  isa_check.py --report DIR [--filter REGEX]

BUILD_DIR is any configured SYCL build of this tree (JIT or AOT; configuring is enough, CMake writes
compile_commands.json). FILE is a file under ggml/src/ggml-sycl, e.g. fattn-dpas.cpp. The script takes FILE's
compile command from that build, swaps its SYCL targets for spir64_gen with the device options
GGML_SYCL_DEVICE_ARCH uses, sets GGML_SYCL_XMX_AOT_SG to the device's DPAS width (8 on Arc A-series, 16
elsewhere), and links a throwaway shared object so ocloc lowers every kernel. IGC's shader dumps go to
DIR/dump. One line per kernel follows:

  GRF     register file the kernel got (128 or 256)
  spill   bytes spilled per thread (0 is the goal)
  dpas    DPAS instructions in the kernel's code (static count, not executions)
  dummy   dummy DPAS IGC inserted for Wa_18023229625 (Arc A-series): an 8x8 DPAS with a null src0 whose
          destination and both sources are one register. IGC adds one before the first DPAS of a basic
          block and before a back-to-back 8x8 DPAS that would reuse src1. It takes an issue slot like a
          real DPAS. A kernel with about one dummy per DPAS has a branch around each DPAS, or consecutive
          DPAS sharing src1 (order them to share src2 and vary src1 instead).

Counts say what the compiler produced, not how fast it runs: decide speed with ../ab/.
Needs the oneAPI compiler on PATH (source setvars.sh) and c++filt for readable kernel names.
"""

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

DPAS = re.compile(r'^\s*(?:\([^)]*\)\s*)?dpas\.(\d+x\d+)\s+\((\d+)\|M\d+\)\s+(r\d+)[.:]\S*\s+(\S+)\s+(r\d+)[.:]\S*\s+(r\d+)[.:]\S*')
AOT_OPTIONS = "-ze-intel-greater-than-4GB-buffer-required -ze-opt-enable-auto-large-GRF-mode"


def dpas_width(device):
    return 8 if device.startswith(("acm", "dg2")) else 16


def find_entry(build_dir, name):
    path = os.path.join(build_dir, "compile_commands.json")
    if not os.path.exists(path):
        sys.exit(f"isa_check: no {path} (configure the build first)")
    with open(path) as f:
        entries = json.load(f)
    want = "/ggml-sycl/" + os.path.basename(name)
    hits = [e for e in entries if e["file"].replace(os.sep, "/").endswith(want)]
    if not hits:
        sys.exit(f"isa_check: {name} is not in {path}")
    return hits[0]


def aot_command(entry, device, extra_defs):
    args = entry.get("arguments") or shlex.split(entry["command"])
    src = os.path.normpath(os.path.join(entry["directory"], entry["file"]))
    out = []
    i = 1
    while i < len(args):
        a = args[i]
        if a in ("-o", "-MF", "-MT", "-MQ", "-Xs", "-Xsycl-target-backend") or a.startswith("-Xsycl-target-backend="):
            i += 2  # the flag and its value
            continue
        if a in ("-c", "-MD", "-MMD") or a.startswith(("-fsycl-targets=", "-DGGML_SYCL_XMX_AOT_SG=", "-flto",
                                                       "-ffat-lto-objects")):
            i += 1
            continue
        if os.path.normpath(os.path.join(entry["directory"], a)) == src:
            i += 1
            continue
        out.append(a)
        i += 1
    targets = ["-fsycl-targets=spir64_gen", "-Xsycl-target-backend=spir64_gen",
               f"-device {device} -options '{AOT_OPTIONS}'"]
    defs = [f"-DGGML_SYCL_XMX_AOT_SG={dpas_width(device)}"] + [f"-D{d}" for d in extra_defs]
    return args[0], out + targets + defs, src, targets


def compile_tu(a):
    entry = find_entry(a.build, a.file)
    cxx, flags, src, targets = aot_command(entry, a.device, a.define or [])
    stem = os.path.splitext(os.path.basename(src))[0]
    out = os.path.abspath(a.out or f"isa-{stem}-{a.device}")
    dump = os.path.join(out, "dump")
    if os.path.isdir(dump) and os.listdir(dump):
        sys.exit(f"isa_check: {dump} already holds dumps; pass a new --out")
    os.makedirs(dump, exist_ok=True)
    env = dict(os.environ, IGC_ShaderDumpEnable="1", IGC_DumpToCustomDir=dump)
    obj, so = os.path.join(out, stem + ".o"), os.path.join(out, stem + ".so")
    steps = [("compile", [cxx] + flags + ["-c", src, "-o", obj]),
             ("device link", [cxx, "-fsycl"] + targets + ["-shared", "-fPIC", obj, "-o", so])]
    for what, cmd in steps:
        log = os.path.join(out, what.replace(" ", "-") + ".log")
        with open(log, "w") as f:
            f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=entry["directory"], env=env, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            sys.exit(f"isa_check: {what} failed for {a.device}, see {log}")
    print(f"{os.path.basename(src)} for {a.device} (DPAS width {dpas_width(a.device)}): dumps in {dump}")
    return out


def short_name(mangled):
    if shutil.which("c++filt"):
        dem = subprocess.run(["c++filt", mangled], capture_output=True, text=True).stdout.strip()
        m = re.search(r"([A-Za-z_]\w*<[^()]*>)\(", dem) or re.search(r"([A-Za-z_]\w*)\(", dem)
        if m:
            return m.group(1)
    return mangled


def report(out, flt):
    # IGC may compile a kernel more than once (another register file, another SIMD width): keep the last dump
    last = {}
    for f in sorted(glob.glob(os.path.join(out, "dump", "*.asm")), key=os.path.getmtime):
        text = open(f, errors="replace").read()
        m = re.search(r"^//\.kernel (\S+)", text, re.M)
        if m and not m.group(1).startswith("Intel_Symbol_Table"):
            last[m.group(1)] = text
    rows = []
    for kernel, text in last.items():
        name = short_name(kernel)
        if flt and not re.search(flt, name) and not re.search(flt, kernel):
            continue
        grf = re.search(r"numGRF=(\d+)", text)
        spill = re.search(r"^//\.spill size (\d+)", text, re.M)
        real = dummy = 0
        for line in text.splitlines():
            d = DPAS.match(line)
            if not d:
                continue
            _, _, dst, src0, src1, src2 = d.groups()
            if src0.startswith("null") and dst == src1 == src2 and "Atomic" in line:
                dummy += 1
            else:
                real += 1
        rows.append((name, grf.group(1) if grf else "?", int(spill.group(1)) if spill else 0, real, dummy))
    rows.sort()
    print(f"{'GRF':>4} {'spill':>6} {'dpas':>5} {'dummy':>5}  kernel")
    for name, grf, spill, real, dummy in rows:
        print(f"{grf:>4} {spill:>6} {real:>5} {dummy:>5}  {name}")
    print(f"{len(rows)} kernels, {sum(r[2] > 0 for r in rows)} spill ({sum(r[2] for r in rows)} B), "
          f"{sum(r[3] for r in rows)} DPAS, {sum(r[4] for r in rows)} dummy DPAS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build", nargs="?", help="configured build directory (holds compile_commands.json)")
    ap.add_argument("file", nargs="?", help="source file under ggml/src/ggml-sycl, e.g. fattn-dpas.cpp")
    ap.add_argument("--device", default="acm-g10", help="ocloc device (default acm-g10, the A770)")
    ap.add_argument("--out", help="output directory (default isa-<file>-<device>)")
    ap.add_argument("--filter", help="only kernels whose name matches this regex")
    ap.add_argument("-D", dest="define", action="append", help="extra preprocessor definition, NAME[=VALUE]")
    ap.add_argument("--report", metavar="DIR", help="only print the table for an earlier run")
    a = ap.parse_args()
    if a.report:
        report(a.report, a.filter)
        return
    if not a.build or not a.file:
        ap.error("BUILD_DIR and FILE are required unless --report is given")
    report(compile_tu(a), a.filter)


if __name__ == "__main__":
    main()
