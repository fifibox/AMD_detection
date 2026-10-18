"""run_all.py -- runs lines of experiments.txt (called by job.slurm, one line per array task).

Builds the train_cv.py command for each line and SKIPS anything already finished
(summary.json present; finished folds are reused via --resume), so a job that hit
its walltime or was pre-empted can simply be resubmitted.

    python run_all.py --data_root /group/pmc078/fwei/DATASET --checkpoint .../RETFound_mae_natureCFP.pth \
        --runs /group/pmc078/fwei/thesis_v2_hpc/runs --amp none --lines 3
    python run_all.py ... --dry_run           # print every command, run nothing
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_experiments(path):
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    out = []
    for i, parts in enumerate(lines):
        task, arch, strategy, k, seed = parts[:5]
        suffix = parts[5] if len(parts) > 5 else ""
        extra = parts[6:]
        tag = f"{arch}_{strategy}" + (k if strategy == "partial" else "")
        if suffix and suffix != "-":
            tag += f"_{suffix}"
        out.append(dict(idx=i, task=task, arch=arch, strategy=strategy, k=k, seed=seed, tag=tag, extra=extra))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--runs", required=True, help="output root (on /group, persistent)")
    ap.add_argument("--experiments", default=str(HERE.parent / "config" / "experiments.txt"))
    ap.add_argument("--lines", type=int, nargs="*", help="subset of line indices (default: all)")
    ap.add_argument("--amp", default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    exps = read_experiments(a.experiments)
    if a.lines:
        exps = [e for e in exps if e["idx"] in set(a.lines)]
    logs = Path(a.runs) / "_logs"
    logs.mkdir(parents=True, exist_ok=True)

    for e in exps:
        out = Path(a.runs) / e["task"] / e["tag"] / f"seed{e['seed']}"
        name = f"[{e['idx']:02d}] {e['task']} {e['tag']} seed{e['seed']}"
        if (out / "summary.json").exists():
            print(f"{name}: done, skipping")
            continue
        cmd = [sys.executable, str(HERE / "train_cv.py"),
               "--data_root", a.data_root, "--task", e["task"], "--arch", e["arch"],
               "--strategy", e["strategy"], "--k", e["k"], "--checkpoint", a.checkpoint,
               "--seed", e["seed"], "--n_splits", "5", "--output_dir", str(out),
               "--amp", a.amp, "--resume", *e["extra"]]
        if a.num_workers is not None:
            cmd += ["--num_workers", str(a.num_workers)]
        print(f"\n{'=' * 80}\n{name}\n  {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        log = logs / f"{e['idx']:02d}_{e['task']}_{e['tag']}_seed{e['seed']}.log"
        with open(log, "a") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                f.write(line)
                # keep the SLURM .out file short: per-epoch lines go to runs/_logs only
                if not line.lstrip().startswith("ep "):
                    print(line, end="", flush=True)
            rc = p.wait()
        print(f"{name}: exit {rc} after {(time.time() - t0) / 60:.1f} min (log: {log})", flush=True)
        if rc != 0:
            sys.exit(f"Stopped: {name} failed. Fix the error and re-run; finished folds are kept.")


if __name__ == "__main__":
    main()
