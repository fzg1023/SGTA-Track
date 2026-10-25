from rgbt import LasHeR
import os

tag = "lasher_train_10"
result_path = os.environ.get(
    "RESULT_PATH",
    "./RGBT_workspace/results/lasher/lasher_train_10",
)
bench = LasHeR()
bench(tracker_name=tag, result_path=result_path, bbox_type="ltwh")
seqs = bench.seqs_name
print("n seqs:", len(seqs))

bad = []
for name in seqs:
    try:
        gt = bench[name]
    except Exception:
        gt = bench[name]["visible"]
    serial = bench.trackers[tag][name]
    if len(serial) < len(gt):
        bad.append((name, len(serial), len(gt)))
    # check row widths
    for i, row in enumerate(serial):
        if len(row) < 4:
            bad.append((name, "short-row", i, len(row), row))
print("seqs with serial<gt:", len(bad))
for b in bad[:10]:
    print(b)
