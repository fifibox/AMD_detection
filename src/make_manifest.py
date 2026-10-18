"""make_manifest.py -- dataset description for thesis Section 3.2 / Table 3.3.

Reads DICOM headers only (no pixels) and writes:
    manifest.csv   one row per image: path, patient, eye, label, date, device...
    manifest_summary.txt  counts by class (images / patients / eyes),
                          eyes per patient, images per eye, devices, dates

    python make_manifest.py --data_root /group/pmc078/fwei/DATASET --out manifest
"""
import argparse
import collections
import csv
from pathlib import Path

import pydicom

from data import TASKS, list_dataset, subject_id_from_path

ORIG = ["No AMD", "Early", "Intermediate", "Late"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="manifest")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    samples, _ = list_dataset(a.data_root, "four_class")
    rows = []
    for path, _, orig in samples:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        rows.append({
            "path": path, "patient": subject_id_from_path(path), "orig_label": orig,
            "binary": TASKS["binary"]["mapping"][orig], "three_class": TASKS["three_class"]["mapping"][orig],
            "eye": str(ds.get("ImageLaterality", "") or ds.get("Laterality", "")),
            "date": str(ds.get("StudyDate", "")), "sex": str(ds.get("PatientSex", "")),
            "model": str(ds.get("ManufacturerModelName", "")), "serial": str(ds.get("DeviceSerialNumber", "")),
            "rows": int(ds.Rows), "cols": int(ds.Columns),
            "photometric": str(ds.get("PhotometricInterpretation", "")),
        })
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    L = []
    p = lambda *s: L.append(" ".join(str(x) for x in s))
    pats = {r["patient"] for r in rows}
    eyes = {(r["patient"], r["eye"]) for r in rows}
    p(f"Images {len(rows)} | patients {len(pats)} | eyes {len(eyes)}")
    p("Missing laterality:", sum(r["eye"] not in ("L", "R") for r in rows))
    for col, names in (("orig_label", ORIG), ("binary", TASKS["binary"]["class_names"]),
                       ("three_class", TASKS["three_class"]["class_names"])):
        p(f"\nBy {col}: class | images | eyes | patients | % images")
        for c, name in enumerate(names):
            sub = [r for r in rows if r[col] == c]
            p(f"  {name:15s} {len(sub):4d} {len({(r['patient'], r['eye']) for r in sub}):4d} "
              f"{len({r['patient'] for r in sub}):4d} {100 * len(sub) / len(rows):5.1f}")
        multi = collections.defaultdict(set)
        for r in rows:
            multi[r["patient"]].add(r[col])
        p(f"  patients with images in >1 {col} class: {sum(len(v) > 1 for v in multi.values())}")
        # an eye with images in two classes is a grading inconsistency
        eye_lab = collections.defaultdict(set)
        for r in rows:
            eye_lab[(r["patient"], r["eye"])].add(r[col])
        bad = [k for k, v in eye_lab.items() if len(v) > 1]
        p(f"  EYES with conflicting {col} labels: {len(bad)} {bad}")

    ep = collections.Counter(len({r['eye'] for r in rows if r['patient'] == q}) for q in pats)
    ie = collections.Counter(collections.Counter((r["patient"], r["eye"]) for r in rows).values())
    p("\nEyes per patient:", dict(ep))
    p("Images per eye:", dict(sorted(ie.items())))
    p("Images per patient:", dict(sorted(collections.Counter(collections.Counter(r['patient'] for r in rows).values()).items())))
    p("Dates per patient >1:", sum(len({r['date'] for r in rows if r['patient'] == q}) > 1 for q in pats))
    p("Date range:", min(r["date"] for r in rows), "-", max(r["date"] for r in rows))
    p("Devices (model, serial, rows x cols):")
    for k, v in sorted(collections.Counter((r["model"], r["serial"], f"{r['rows']}x{r['cols']}") for r in rows).items()):
        p("  ", k, v)
    p("Sex (patients):", dict(collections.Counter({r["patient"]: r["sex"] for r in rows}.values())))
    (out / "manifest_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
