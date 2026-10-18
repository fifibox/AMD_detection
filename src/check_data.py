"""check_data.py -- dataset count + colour sanity check (notebook cells 5-6, without the plot).

    python check_data.py --data_root /group/pmc078/fwei/DATASET --expected 337
Exits non-zero if the image count is wrong or the colours look like a YBR->RGB error.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_dataset, load_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--expected", type=int, default=337)
    a = ap.parse_args()

    for task in ("binary", "three_class", "four_class"):
        s, _ = list_dataset(a.data_root, task)
    if len(s) != a.expected:
        sys.exit(f"found {len(s)} images, expected {a.expected} -- check --data_root")

    def stats(p):
        arr = np.asarray(load_rgb(p))
        m = arr.reshape(-1, 3)[::97]
        return p, arr.shape, m[m.sum(1) > 60].mean(0)

    with ThreadPoolExecutor(os.cpu_count()) as ex:  # also fills AMD_CACHE_DIR if set
        res = list(ex.map(stats, [p for p, _, _ in s]))
    means = np.array([r[2] for r in res])
    bad = [r[0] for r in res if not (r[2][0] > r[2][1] > r[2][2])]
    print("mean RGB over fundus pixels:", means.mean(0).round(1), "| shapes:", sorted({r[1] for r in res}))
    print(f"images not ordered R>G>B: {len(bad)} / {len(res)}")
    if len(bad) >= 0.05 * len(res):
        sys.exit("colour looks wrong (YBR->RGB?) -- stop and check before training")
    print("data check OK")


if __name__ == "__main__":
    main()
