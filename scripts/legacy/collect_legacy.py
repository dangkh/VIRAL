"""
Per-seed results of the original jepa2 code (with / without JEPA loss).

    python scripts/legacy/collect_legacy.py

Reads log/*.log of every worktree next to this repo whose name contains "jepa2"
(VIRAL_jepa2, VIRAL_jepa2_noloss, legacy_jepa2_*). The old quick_start's final BEST block
picks the seed with the highest TEST score, so this script reads the per-seed
"test result:" line written after each seed instead.
"""
import glob
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # parent of the repo
METRIC = re.compile(r'([A-Za-z]+@\d+):\s*([0-9.]+)')
COMBO = re.compile(r'=========(\d+)/(\d+): Parameters:\[.*?\]=\((.*)\)=======')
SHOW = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']

rows = []
for wt in sorted(glob.glob(os.path.join(BASE, '*jepa2*'))):
    if not os.path.isdir(os.path.join(wt, 'log')):
        continue
    variant = 'noloss' if 'noloss' in os.path.basename(wt) else 'loss'
    for path in sorted(glob.glob(os.path.join(wt, 'log', 'VLIF-baby-*.log'))):
        seed, epoch = None, None
        for line in open(path, encoding='utf-8', errors='ignore'):
            m = COMBO.search(line)
            if m:
                seed = m.group(3).split(',')[-1].strip()
                epoch = None
                continue
            m = re.search(r'best eval result in epoch (\d+)', line)
            if m:
                epoch = int(m.group(1))
            if 'INFO test result: ' in line and seed is not None:
                met = {k.lower(): float(v) for k, v in METRIC.findall(line)}
                if met:
                    rows.append(dict(variant=variant, seed=seed, epoch=epoch,
                                     worktree=os.path.basename(wt), log=os.path.basename(path), **met))
                    seed = None

if not rows:
    print('No finished seeds found under', BASE)
    raise SystemExit

print('{:<8} {:>5} {:>6}  '.format('variant', 'seed', 'epoch') + '  '.join('{:>9}'.format(m) for m in SHOW) + '  worktree')
for r in sorted(rows, key=lambda r: (r['variant'], r['seed'], r['worktree'])):
    print('{:<8} {:>5} {:>6}  '.format(r['variant'], r['seed'], str(r['epoch'])) +
          '  '.join('{:>9.4f}'.format(r.get(m, float('nan'))) for m in SHOW) + '  ' + r['worktree'])

print('\nMean +/- std over runs')
for v in ['loss', 'noloss']:
    rs = [r for r in rows if r['variant'] == v]
    if not rs:
        continue
    cells = []
    for m in SHOW:
        xs = [r[m] for r in rs if m in r]
        sd = st.stdev(xs) if len(xs) > 1 else 0.0
        cells.append('{:.4f}+/-{:.4f}'.format(st.mean(xs), sd))
    print('{:<8} n={:<3} '.format(v, len(rs)) + '  '.join(cells))

# same-seed repeats (GPU non-determinism)
print('\nSame-seed repeats (spread = max - min)')
for v in ['loss', 'noloss']:
    by_seed = {}
    for r in rows:
        if r['variant'] == v:
            by_seed.setdefault(r['seed'], []).append(r)
    for s, rs in by_seed.items():
        if len(rs) > 1:
            print('{:<8} seed {:>5}: '.format(v, s) + '  '.join(
                '{} {:.4f}'.format(m, max(r[m] for r in rs) - min(r[m] for r in rs)) for m in SHOW))
