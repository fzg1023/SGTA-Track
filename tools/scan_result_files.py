import os
from rgbt import LasHeR

tags = {
    "lasher_train_10": "./RGBT_workspace/results/lasher/lasher_train_10",
    "lasher_train_15": "./RGBT_workspace/results/lasher/lasher_train_15",
}
bench = LasHeR()
seqs = list(bench.seqs_name)
gt_lens = {}
for name in seqs:
    try:
        gt = bench[name]
    except Exception:
        gt = bench[name]["visible"]
    gt_lens[name] = len(gt)

for tag, path in tags.items():
    print("== %s ==" % tag)
    n_bad = 0
    for name in seqs:
        f = os.path.join(path, name + ".txt")
        if not os.path.isfile(f):
            print("  MISSING:", name)
            n_bad += 1
            continue
        with open(f) as fh:
            rows = [ln.split() for ln in fh if ln.strip()]
        if len(rows) < gt_lens[name]:
            print("  SHORT: %s  rows=%d gt=%d  mtime=%s"
                  % (name, len(rows), gt_lens[name],
                     os.path.getmtime(f)))
            n_bad += 1
    if n_bad == 0:
        print("  all 245 files complete")
