Thesis pipeline: RETFound vs ImageNet-based lightweight models for AMD screening, patient-grouped 5-fold CV, seed 43.

## Layout

- `config/` – `config.sh` (paths, the only file to edit), `experiments.txt` (29 runs)
- `src/` – Python code
- `slurm/` – `job.slurm` (training), `job_analysis.slurm` (analysis)
- `scripts/` – `setup_env.sh`, `submit_all.sh`

## Usage (from the project root)

```bash
bash scripts/setup_env.sh                               # once, on the login node
sbatch --array=0 --export=ALL,SMOKE=1 slurm/job.slurm   # smoke test
bash scripts/submit_all.sh                              # all runs, then analysis
```

Resubmitting skips finished runs and folds. Copy `cascade.py` into `src/` before running the analysis.
