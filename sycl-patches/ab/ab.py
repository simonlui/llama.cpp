#!/usr/bin/env python3
"""A/B runner for the Arc A770 patch-stack experiments in plan.json (see README.md).

  ab.py list   [--plan plan.json] [--tier N]                      what would run, and what is skipped
  ab.py run    [--plan plan.json] [--only a,b] [--tier N] [--rounds R] [--out results.jsonl] [--guard]
  ab.py report results.jsonl [--min-gain 1.0]                     verdict table (markdown)
  ab.py kld    [--plan plan.json] [--only a,b] [--out kld.jsonl]  accuracy of both arms against CPU logits
  ab.py mask   NAME[,NAME...]                                     GGML_SYCL_LOCAL_OFF value for those bits

Each experiment has two arms, A (the baseline: the patch off) and B (the patch on). An arm is an environment
and optionally another binary (for build-flag A/Bs). Rounds alternate the order, A B, B A, A B, ..., so slow drift
(clocks, temperature, background load) hits both arms alike, and the report pairs the two arms within each round:
the gain of a round is B / A - 1, and the verdict comes from the mean of those ratios and its 95% confidence
interval across rounds. Model paths come from environment variables named in the plan (AB_DENSE_Q4_K, ...);
an experiment whose variable is unset is skipped, not failed.
"""

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# GGML_SYCL_LOCAL_OFF bits (ZZ-LOCAL-0018). Keep in sync with ggml_sycl_local_off_bit in common.hpp.
LOCAL_OFF = {
    "iq_ncols": 1 << 0,
    "moe_reorder": 1 << 1,
    "iq4_nl_reorder": 1 << 2,
    "dg2_glu_one_row": 1 << 3,
    "rope_set_rows": 1 << 4,
    "norm_rope": 1 << 5,
    "moe_sum": 1 << 6,
    "multi_add": 1 << 7,
    "topk_moe_ext": 1 << 8,
    "moe_mmvq_fuse": 1 << 9,
    "mul_mat_add": 1 << 10,
    "act_reuse": 1 << 11,
    "model_fusions": 1 << 12,
    "mmvq_add": 1 << 13,
    "iq4_xs_reorder": 1 << 14,
}

# two-sided 95% Student t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df):
    keys = sorted(T95)
    for k in keys:
        if df <= k:
            return T95[k]
    return 1.96


def local_off_mask(names):
    mask = 0
    for n in names:
        n = n.strip()
        if not n:
            continue
        if n == "all":
            mask |= sum(LOCAL_OFF.values())
            continue
        if n not in LOCAL_OFF:
            sys.exit(f"ab.py: unknown GGML_SYCL_LOCAL_OFF name '{n}' (known: {', '.join(LOCAL_OFF)})")
        mask |= LOCAL_OFF[n]
    return mask


def expand(s):
    return os.path.expandvars(os.path.expanduser(s))


def load_plan(path):
    with open(path) as f:
        plan = json.load(f)
    return plan.get("defaults", {}), plan["experiments"]


def arm_env(defaults, exp, arm):
    env = dict(defaults.get("env", {}))
    env.update(exp.get("env", {}))
    env.update(arm.get("env", {}))
    out = {}
    for k, v in env.items():
        if k == "GGML_SYCL_LOCAL_OFF" and isinstance(v, list):
            v = local_off_mask(v)
        out[k] = expand(str(v))
    return out


def arm_bin(defaults, exp, arm):
    return expand(arm.get("bin", exp.get("bin", defaults.get("bin", "${AB_BUILD}/bin/llama-bench"))))


def missing_inputs(defaults, exp):
    """Unset model variables and missing binaries, as a list of reasons (empty: runnable)."""
    reasons = []
    text = " ".join([exp.get("args", ""), exp.get("model", ""), exp.get("bin", defaults.get("bin", "${AB_BUILD}")),
                     exp["a"].get("bin", ""), exp["b"].get("bin", "")])
    for var in sorted(set(re.findall(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", text))):
        if not os.environ.get(var):
            reasons.append(f"${var} unset")
    if reasons:
        return reasons
    for name in ("a", "b"):
        b = arm_bin(defaults, exp, exp[name])
        if not os.path.exists(b):
            reasons.append(f"{name}: no binary {b}")
    return reasons


def selected(experiments, only, tier):
    names = set(only.split(",")) if only else None
    for exp in experiments:
        if names is not None and exp["name"] not in names:
            continue
        if tier is not None and exp.get("tier", 9) > tier:
            continue
        yield exp


SYCL_ENV_LINE = re.compile(r"^\s+(GGML_SYCL_[A-Z0-9_]+): (.*)$")


def run_bench(binary, model, args, env, log_path):
    # -v only adds log lines (to the log file); without it llama-bench silences the backend's settings dump
    cmd = [binary, "-m", model, "-o", "jsonl", "-v"] + shlex.split(expand(args))
    full_env = dict(os.environ)
    full_env.update(env)
    t0 = time.time()
    p = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    with open(log_path, "a") as log:
        log.write(f"### {time.strftime('%F %T')} {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(cmd)}\n")
        log.write(p.stderr)
    # the backend prints its settings at start: record them, so a mistyped variable shows up in the results
    seen = {}
    for line in p.stderr.splitlines():
        m = SYCL_ENV_LINE.match(line)
        if m:
            seen[m.group(1)] = m.group(2).strip()
    rows = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return p.returncode, rows, seen, time.time() - t0


def test_key(row):
    key = f"pp{row['n_prompt']}" if row.get("n_gen", 0) == 0 else (
        f"tg{row['n_gen']}" if row.get("n_prompt", 0) == 0 else f"pp{row['n_prompt']}+tg{row['n_gen']}")
    if row.get("n_depth"):
        key += f"@d{row['n_depth']}"
    if row.get("n_ubatch") and row.get("n_gen", 0) == 0 and row["n_ubatch"] < row.get("n_prompt", 0):
        key += f"/ub{row['n_ubatch']}"
    return key


def guard(wait):
    g = os.path.join(HERE, "guard.sh")
    return subprocess.run(["bash", g, "--wait", str(wait)], capture_output=True, text=True)


def cmd_list(a):
    defaults, experiments = load_plan(a.plan)
    for exp in selected(experiments, a.only, a.tier):
        why = missing_inputs(defaults, exp)
        state = "run " if not why else "skip"
        print(f"{state} T{exp.get('tier', '?')} {exp['name']:<28} {exp.get('patch', ''):<22} {'; '.join(why)}")


def cmd_run(a):
    defaults, experiments = load_plan(a.plan)
    rounds = a.rounds or defaults.get("rounds", 4)
    log_path = os.path.splitext(a.out)[0] + ".log"
    with open(a.out, "a") as out:
        if a.guard:
            st = subprocess.run(["bash", os.path.join(HERE, "guard.sh"), "--stamp"], capture_output=True, text=True)
            out.write(json.dumps({"type": "stamp", "time": time.strftime("%F %T"), "text": st.stdout}) + "\n")
            sys.stderr.write(st.stdout)
        for exp in selected(experiments, a.only, a.tier):
            why = missing_inputs(defaults, exp)
            if why:
                print(f"SKIP {exp['name']}: {'; '.join(why)}", file=sys.stderr)
                continue
            model = expand(exp["model"])
            args = exp.get("args", defaults.get("args", "-p 512 -n 128"))
            reps = exp.get("reps", defaults.get("reps", 3))
            if "-r " not in f" {args} ":
                args = f"{args} -r {reps}"
            r_exp = exp.get("rounds", rounds)
            for r in range(r_exp):
                order = ("a", "b") if r % 2 == 0 else ("b", "a")
                for arm in order:
                    if a.guard:
                        g = guard(defaults.get("guard_wait", 60))
                        if g.returncode != 0:
                            print(f"guard: {g.stderr.strip()}", file=sys.stderr)
                    cd = exp.get("cooldown", defaults.get("cooldown", 5))
                    if cd:
                        time.sleep(cd)
                    env = arm_env(defaults, exp, exp[arm])
                    rc, rows, seen, secs = run_bench(arm_bin(defaults, exp, exp[arm]), model, args, env, log_path)
                    rec = {"type": "run", "exp": exp["name"], "patch": exp.get("patch", ""), "round": r, "arm": arm,
                           "env": env, "rc": rc, "secs": round(secs, 1), "sycl_env": seen,
                           "tests": [{"test": test_key(x), "avg_ts": x.get("avg_ts"),
                                      "samples_ts": x.get("samples_ts", [])} for x in rows]}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    summary = ", ".join(f"{t['test']} {t['avg_ts']:.2f}" for t in rec["tests"] if t["avg_ts"])
                    print(f"{exp['name']} r{r} {arm}: rc={rc} {summary} ({secs:.0f}s)", file=sys.stderr)
                    if rc != 0:
                        print(f"  failed, see {log_path}", file=sys.stderr)


def cmd_report(a):
    runs = {}
    meta = {}
    with open(a.results) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "run" or rec["rc"] != 0:
                continue
            meta[rec["exp"]] = rec.get("patch", "")
            for t in rec["tests"]:
                if t["avg_ts"]:
                    runs.setdefault((rec["exp"], t["test"]), {}).setdefault(rec["round"], {})[rec["arm"]] = t["avg_ts"]
    thr = a.min_gain / 100.0
    print("| experiment | patch | test | A t/s | B t/s | gain | 95% CI | rounds | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (exp, test), by_round in sorted(runs.items()):
        pairs = [(v["a"], v["b"]) for v in by_round.values() if "a" in v and "b" in v]
        if not pairs:
            continue
        gains = [b / x - 1.0 for x, b in pairs]
        n = len(gains)
        mean = sum(gains) / n
        if n > 1:
            sd = math.sqrt(sum((g - mean) ** 2 for g in gains) / (n - 1))
            half = t95(n - 1) * sd / math.sqrt(n)
        else:
            half = float("inf")
        lo, hi = mean - half, mean + half
        if n < 3:
            verdict = "more rounds"
        elif lo > 0 and mean >= thr:
            verdict = "WIN"
        elif hi < 0 and mean <= -thr:
            verdict = "LOSS"
        elif lo > -thr and hi < thr:
            verdict = "no effect"
        else:
            verdict = "inconclusive"
        ma = sum(x for x, _ in pairs) / n
        mb = sum(b for _, b in pairs) / n
        ci = f"{lo * 100:+.2f}..{hi * 100:+.2f}%" if math.isfinite(half) else "n/a"
        print(f"| {exp} | {meta.get(exp, '')} | {test} | {ma:.2f} | {mb:.2f} | {mean * 100:+.2f}% | {ci} | {n} | {verdict} |")


KLD_LINES = {
    "kld": re.compile(r"Mean\s+KLD:\s+([-0-9.eE]+)\s+±\s+([-0-9.eE]+)"),
    "top_p": re.compile(r"Same top p:\s+([-0-9.eE]+)\s+±\s+([-0-9.eE]+)"),
    "ppl": re.compile(r"Mean PPL\(Q\)\s+:\s+([-0-9.eE]+)\s+±\s+([-0-9.eE]+)"),
}


def perplexity_bin(bench_bin):
    return os.path.join(os.path.dirname(bench_bin), "llama-perplexity")


def run_perplexity(binary, model, args, env, log_path):
    cmd = [binary, "-m", model] + shlex.split(expand(args))
    full_env = dict(os.environ)
    full_env.update(env)
    p = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    with open(log_path, "a") as log:
        log.write(f"### {time.strftime('%F %T')} {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(cmd)}\n")
        log.write(p.stdout)
        log.write(p.stderr)
    found = {}
    for key, rx in KLD_LINES.items():
        m = rx.search(p.stdout + p.stderr)
        if m:
            found[key] = (float(m.group(1)), float(m.group(2)))
    return p.returncode, found


def cmd_kld(a):
    """Both arms against logits from the CPU backend (-dev none), so neither arm is its own reference."""
    defaults, experiments = load_plan(a.plan)
    log_path = os.path.splitext(a.out)[0] + ".log"
    base_dir = expand(a.base_dir)
    os.makedirs(base_dir, exist_ok=True)
    print("| experiment | arm | mean KLD | same top p | PPL |")
    print("|---|---|---|---|---|")
    with open(a.out, "a") as out:
        for exp in selected(experiments, a.only, None):
            if "kld" not in exp:
                continue
            why = missing_inputs(defaults, exp)
            if why or not os.environ.get("AB_TEXT"):
                print(f"SKIP {exp['name']}: {'; '.join(why or ['$AB_TEXT unset'])}", file=sys.stderr)
                continue
            model = expand(exp["model"])
            kargs = exp["kld"].get("args", defaults.get("kld_args", "-c 512 --chunks 40"))
            tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{os.path.basename(model)}_{kargs}")
            base = os.path.join(base_dir, tag + ".kld")
            pbin_b = perplexity_bin(arm_bin(defaults, exp, exp["b"]))
            if not os.path.exists(base):
                print(f"{exp['name']}: CPU reference logits -> {base}", file=sys.stderr)
                rc, _ = run_perplexity(pbin_b, model, f"-f {expand('$AB_TEXT')} {kargs} -dev none -ngl 0 "
                                       f"--kl-divergence-base {base}", {}, log_path)
                if rc != 0:
                    print(f"  failed, see {log_path}", file=sys.stderr)
                    continue
            res = {}
            for arm in ("a", "b"):
                env = arm_env(defaults, exp, exp[arm])
                rc, found = run_perplexity(perplexity_bin(arm_bin(defaults, exp, exp[arm])), model,
                                           f"{kargs} -ngl 99 --kl-divergence-base {base} --kl-divergence", env,
                                           log_path)
                res[arm] = found
                out.write(json.dumps({"type": "kld", "exp": exp["name"], "arm": arm, "env": env, "rc": rc,
                                      **{k: v for k, v in found.items()}}) + "\n")
                out.flush()
                kld = found.get("kld", (float("nan"), 0))
                top = found.get("top_p", (float("nan"), 0))
                ppl = found.get("ppl", (float("nan"), 0))
                print(f"| {exp['name']} | {arm} | {kld[0]:.6f} ± {kld[1]:.6f} | {top[0]:.3f} ± {top[1]:.3f}% | "
                      f"{ppl[0]:.4f} |")
            ka, kb = res["a"].get("kld"), res["b"].get("kld")
            ta, tb = res["a"].get("top_p"), res["b"].get("top_p")
            if ka and kb and ta and tb:
                # B may not be measurably less accurate than A: a tenth more KLD plus a floor for runs near 0,
                # and at most 0.3 points fewer top-token matches
                ok = kb[0] <= ka[0] * 1.10 + 2e-4 and tb[0] >= ta[0] - 0.3
                print(f"| {exp['name']} | verdict | {'accuracy OK' if ok else 'ACCURACY LOSS'} | | |")


def cmd_mask(a):
    print(local_off_mask(a.names.split(",")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "run"):
        p = sub.add_parser(name)
        p.add_argument("--plan", default=os.path.join(HERE, "plan.json"))
        p.add_argument("--only")
        p.add_argument("--tier", type=int)
        if name == "run":
            p.add_argument("--rounds", type=int)
            p.add_argument("--out", default="ab-results.jsonl")
            p.add_argument("--guard", action="store_true", help="stamp the setup, wait for an idle GPU between runs")
    p = sub.add_parser("kld")
    p.add_argument("--plan", default=os.path.join(HERE, "plan.json"))
    p.add_argument("--only")
    p.add_argument("--out", default="ab-kld.jsonl")
    p.add_argument("--base-dir", default="ab-kld-base", help="where the CPU reference logits are kept")
    p = sub.add_parser("report")
    p.add_argument("results")
    p.add_argument("--min-gain", type=float, default=1.0, help="smallest gain in percent that counts (default 1)")
    p = sub.add_parser("mask")
    p.add_argument("names")
    a = ap.parse_args()
    {"list": cmd_list, "run": cmd_run, "report": cmd_report, "kld": cmd_kld, "mask": cmd_mask}[a.cmd](a)


if __name__ == "__main__":
    main()
