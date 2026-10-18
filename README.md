# thesis_v2_hpc — Kaya 版本（与 Colab notebook V5 同一套流程）

把整个文件夹放到 `/group/pmc078/fwei/thesis_v2_hpc`。**所有命令都在这个根目录下运行。**

```
thesis_v2_hpc/
├── README.md
├── config/
│   ├── config.sh            ← 唯一需要改的文件：路径、venv、AMP（notebook cell 1）
│   └── experiments.txt      ← 29 行实验（第 0–28 行）
├── src/                     ← 所有 Python 代码（从 notebook 原样提取）
│   ├── data.py  models_vit.py  models.py  evaluate.py
│   ├── train_cv.py  run_all.py
│   ├── aggregate.py  compare.py  make_manifest.py  check_data.py
│   ├── quality_m1.py  quality_analysis.py  quality_validation.py
│   └── (cascade.py)         ← 请从 Drive MyDrive/AMD/cascade.py 拷进来
├── slurm/
│   ├── job.slurm            ← 训练：array 0–28，一行实验 = 一个 array task
│   └── job_analysis.slurm   ← 训练后：表格、配对比较、AutoMorph 质量分析、cascade
├── scripts/
│   ├── setup_env.sh         ← 登录节点跑一次：装包、下权重、clone AutoMorph、检查数据
│   └── submit_all.sh        ← 一次提交训练 + 分析
│
│   （以下运行后自动生成）
├── logs/                    ← SLURM 输出
├── runs/                    ← 训练结果（和 Drive 上 AMD/runs 结构相同），runs/_logs 为逐 epoch 日志
└── results/                 ← manifest/ 和 quality/
```

## 步骤

```bash
cd /group/pmc078/fwei/thesis_v2_hpc
export HF_TOKEN=hf_...                    # 只有权重还没下载时需要
bash scripts/setup_env.sh                 # 登录节点，一次（也会建 logs/）
sbatch --array=0 --export=ALL,SMOKE=1 slurm/job.slurm   # 冒烟测试（几分钟）
bash scripts/submit_all.sh                # 全部 29 个实验 + 分析
```

- 只跑某些行：`sbatch --array=18-28 slurm/job.slurm`
- 超时/被中断：直接重新 `sbatch`，已完成的 run 和 fold 会自动跳过。
- 一定要在根目录提交（不要 `cd slurm` 再提交），否则 `logs/` 和 `config/` 找不到，脚本会报错提示。

## 注意

- `DATA` 必须是 337 张、文件名带 `SUBJ_<id>_` 的重新整理版数据集；不是的话 `check_data.py` 会报错。
- GPU 默认 `gpu` 分区 V100-32GB；用 MI210 按 `slurm/job.slurm` 注释改分区/gres，并装 ROCm 版 PyTorch（代码不用改）。
- `AMP=auto` 与 Colab 规则一致（V100→fp16，A100/MI210→bf16）；fp32 设 `AMP=none`。
