"""
Summarize experiment logs written by src/main.py (one row per experiment).

    python scripts/collect_results.py              # dataset baby, logs in ./log
    python scripts/collect_results.py -d sports
    python scripts/collect_results.py --all        # keep every run, not only the latest per name

Writes results_<dataset>.csv next to the log folder and prints a table sorted by test Recall@20.
"""
import argparse
import csv
import glob
import os
import re
from datetime import datetime

TIME_RE = r'[A-Z][a-z]{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2}'
METRIC_RE = re.compile(r'([A-Za-z]+@\d+):\s*([0-9.]+)')
VPID_RE = re.compile(r'\b(R|U_v|U_t|S|R2_vt|cf_only)=(-?[0-9.]+)')
SHOW = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']


def parse_metrics(line):
    return {k.lower(): float(v) for k, v in METRIC_RE.findall(line)}


def parse_log(path):
    text = open(path, encoding='utf-8', errors='ignore').read()
    row = {'status': 'running'}

    final = text.rfind('BEST ████')
    current = text.rfind('Current BEST')
    block = None
    if final >= 0:
        row['status'] = 'done'
        block = text[final:]
    elif current >= 0:
        block = text[current:]
    if block is not None:
        v = re.search(r'Valid:\s*(.*)', block)
        t = re.search(r'Test:\s*(.*)', block)
        if v:
            row.update({'valid_' + k: x for k, x in parse_metrics(v.group(1)).items()})
        if t:
            row.update({'test_' + k: x for k, x in parse_metrics(t.group(1)).items()})
    else:
        # no finished hyper-parameter combo yet: best test so far from per-epoch logs
        if 'Training loss is nan' in text or 'Traceback' in text:
            row['status'] = 'failed'

    epochs = re.findall(r'epoch (\d+) training', text)
    row['epochs'] = int(epochs[-1]) + 1 if epochs else 0
    best_ep = re.findall(r'best eval result in epoch (\d+)', text)
    row['best_epoch'] = int(best_ep[-1]) if best_ep else ''

    held = [ln for ln in text.splitlines() if '[V-PID held-out-items]' in ln]
    if held:
        row.update({'vpid_' + k: float(x) for k, x in VPID_RE.findall(held[-1])})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--log_dir', default='log')
    ap.add_argument('--model', default='VLIF')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()

    name_re = re.compile(r'^{}-{}-(?P<name>.+)-(?P<t>{})\.log$'.format(
        re.escape(args.model), re.escape(args.dataset), TIME_RE))
    runs = []
    for path in glob.glob(os.path.join(args.log_dir, '{}-{}-*.log'.format(args.model, args.dataset))):
        m = name_re.match(os.path.basename(path))
        if not m:
            continue  # logs without --exp_name
        row = {'name': m.group('name'),
               'time': datetime.strptime(m.group('t'), '%b-%d-%Y-%H-%M-%S'),
               'log': path}
        row.update(parse_log(path))
        runs.append(row)

    if not args.all:
        latest = {}
        for r in runs:
            if r['name'] not in latest or r['time'] > latest[r['name']]['time']:
                latest[r['name']] = r
        runs = list(latest.values())
    if not runs:
        print('No logs with an experiment name found in', args.log_dir)
        return

    runs.sort(key=lambda r: r.get('test_recall@20', -1), reverse=True)

    vpid_cols = ['vpid_R', 'vpid_U_v', 'vpid_U_t', 'vpid_S', 'vpid_R2_vt']
    head = ['name', 'status', 'epochs', 'best_epoch', 'valid_recall@20'] + ['test_' + m for m in SHOW] + vpid_cols
    fmt = lambda x: '{:.4f}'.format(x) if isinstance(x, float) else str(x)
    widths = [max(len(h), *(len(fmt(r.get(h, ''))) for r in runs)) for h in head]
    print('  '.join(h.ljust(w) for h, w in zip(head, widths)))
    print('  '.join('-' * w for w in widths))
    for r in runs:
        print('  '.join(fmt(r.get(h, '')).ljust(w) for h, w in zip(head, widths)))

    out = os.path.join(os.path.dirname(os.path.abspath(args.log_dir)), 'results_{}.csv'.format(args.dataset))
    keys = head + sorted({k for r in runs for k in r} - set(head) - {'time'}) + ['time']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, '') for k in keys})
    print('\nSaved', out)


if __name__ == '__main__':
    main()
