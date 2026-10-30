# coding: utf-8
"""
Linear probe for "collaborative signal Y as the latent z of a cross-modal JEPA".

All quantities are held-out (K-fold over items) linear-Gaussian V-information
    I_V(X -> T) = sum_d 0.5 * log( Var(T_d) / MSE_d(X) )     [nats]
with ridge regression as the predictor family (alpha picked by CV), plus R^2.

  A. PID about Y (target = collaborative item embedding Y), linear commonality analysis
     on held-out explained variance (R^2), additive for a linear predictor family:
       U_v = R2(vt) - R2(t),  U_t = R2(vt) - R2(v),  R = R2(v) + R2(t) - R2(vt)
     (R < 0 means a suppression / synergy-like effect). The MMI rule R = min(I_v, I_t) is
     NOT used: it always sets the weaker modality's uniqueness to zero.
  B. Latent-z JEPA view (target = the other modality's embedding s_v / s_t):
       gain_v = I([t,Y] -> s_v) - I(t -> s_v)   ~ what Y adds when text predicts image  (~ U_v + S)
       gain_t = I([v,Y] -> s_t) - I(v -> s_t)   ~ what Y adds when image predicts text  (~ U_t + S)
     control: the same gain with Y randomly permuted over items (should be ~0)
  C. Popularity: how much of Y (and of v, t) is explained by log(1 + train degree).
Everything is reported for all items and per popularity group (head / mid / tail,
same definition as pop_eval in the evaluator). With --debias the probes are repeated
on a popularity-debiased Y (log-degree regressed out).

Run from the repo root (numpy / pandas / pyyaml only, CPU, ~1 min on Baby):
    python src/probe_latent_z.py -d baby
    python src/probe_latent_z.py -d baby --v_file image_feat.npy --t_file text_feat.npy --debias
Output: printed tables + log/probe_latent_z_<dataset>[_<tag>].json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

GROUPS = ('all', 'head', 'mid', 'tail')


# ------------------------------------------------------------------ data
def load_config(dataset):
    cfg = {}
    for f in ['src/configs/overall.yaml', 'src/configs/dataset/{}.yaml'.format(dataset)]:
        if os.path.isfile(f):
            with open(f, encoding='utf-8') as fh:
                cfg.update(yaml.safe_load(fh) or {})
    return cfg


def load_data(args):
    cfg = load_config(args.dataset)
    root = os.path.join(cfg.get('data_path', 'data/'), args.dataset)
    uid, iid = cfg['USER_ID_FIELD'], cfg['ITEM_ID_FIELD']
    split = cfg.get('inter_splitting_label', 'x_label')
    df = pd.read_csv(os.path.join(root, cfg['inter_file_name']), sep=cfg.get('field_separator', '\t'),
                     usecols=[uid, iid, split])
    n_items = int(df[iid].max()) + 1
    degree = np.bincount(df.loc[df[split] == 0, iid].values.astype(np.int64), minlength=n_items)

    v = np.load(os.path.join(root, args.v_file or cfg['vision_feature_file']), allow_pickle=True).astype(np.float64)
    t = np.load(os.path.join(root, args.t_file or cfg['text_feature_file']), allow_pickle=True).astype(np.float64)
    y = np.load(os.path.join(root, args.y_file), allow_pickle=True).astype(np.float64)
    for name, a in [('v', v), ('t', t), ('Y', y)]:
        assert a.shape[0] == n_items, '{} has {} rows, expected n_items={}'.format(name, a.shape[0], n_items)
    return v, t, y, degree, n_items


def popularity_groups(degree, cuts):
    n = len(degree)
    order = np.argsort(-degree, kind='stable')
    group = np.full(n, 2, dtype=np.int64)
    group[order[:int(np.ceil(cuts[1] * n))]] = 1
    group[order[:int(np.ceil(cuts[0] * n))]] = 0
    return group


# ------------------------------------------------------------------ linear algebra
def standardize(a):
    a = a - a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True)
    return a / np.where(sd > 1e-12, sd, 1.0)


def pca(a, k):
    a = a - a.mean(0, keepdims=True)
    if k is None or k >= min(a.shape):
        return a
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, :k] * s[:k]


def kfold(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def ridge_oof(x, y, folds, alphas):
    """Out-of-fold ridge predictions for every alpha: returns {alpha: pred[n, d]}."""
    preds = {a: np.zeros_like(y) for a in alphas}
    n = x.shape[0]
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te, assume_unique=False)
        mx, my = x[tr].mean(0), y[tr].mean(0)
        u, s, vt = np.linalg.svd(x[tr] - mx, full_matrices=False)
        uty = u.T @ (y[tr] - my)
        xte = (x[te] - mx) @ vt.T
        for a in alphas:
            preds[a][te] = (xte * (s / (s ** 2 + a))) @ uty + my
    return preds


def scores(pred, target, mask):
    t, p = target[mask], pred[mask]
    var = t.var(0)
    mse = ((p - t) ** 2).mean(0)
    keep = var > 1e-12
    if not keep.any():
        return float('nan'), float('nan')
    info = float((0.5 * np.log(var[keep] / np.maximum(mse[keep], 1e-12))).sum())
    r2 = float(1.0 - mse[keep].sum() / var[keep].sum())
    return info, r2


class Probe:
    def __init__(self, folds, alphas, masks):
        self.folds, self.alphas, self.masks = folds, alphas, masks

    def __call__(self, x, target):
        """Best-alpha (by overall held-out R^2) out-of-fold scores per group."""
        preds = ridge_oof(x, target, self.folds, self.alphas)
        best = max(self.alphas, key=lambda a: scores(preds[a], target, self.masks['all'])[1])
        res = {g: dict(zip(('info', 'r2'), scores(preds[best], target, m))) for g, m in self.masks.items()}
        res['alpha'] = best
        return res


# ------------------------------------------------------------------ analysis
def run_all(v, t, y, degree, group, args, tag):
    rng = np.random.default_rng(args.seed)
    keep = degree >= args.min_degree
    v, t, y, degree, group = v[keep], t[keep], y[keep], degree[keep], group[keep]
    n = len(y)
    masks = {'all': np.ones(n, bool), 'head': group == 0, 'mid': group == 1, 'tail': group == 2}
    folds = kfold(n, args.folds, rng)
    probe = Probe(folds, args.alphas, masks)

    xv = standardize(pca(standardize(v), args.pca))         # inputs
    xt = standardize(pca(standardize(t), args.pca))
    sv = standardize(pca(standardize(v), args.target_dim))  # JEPA targets s_v, s_t
    st_ = standardize(pca(standardize(t), args.target_dim))
    ys = standardize(y)
    y_perm = ys[rng.permutation(n)]
    logdeg = standardize(np.log1p(degree)[:, None].astype(np.float64))

    out = {'tag': tag, 'n_items': int(n), 'group_sizes': {g: int(m.sum()) for g, m in masks.items()}}

    # A. PID about Y
    a = {'v->Y': probe(xv, ys), 't->Y': probe(xt, ys), 'vt->Y': probe(np.hstack([xv, xt]), ys)}
    pid = {}
    for g in GROUPS:
        rv, rt, rvt = a['v->Y'][g]['r2'], a['t->Y'][g]['r2'], a['vt->Y'][g]['r2']
        pid[g] = {'R': rv + rt - rvt, 'U_v': rvt - rt, 'U_t': rvt - rv, 'unexplained': 1.0 - rvt}
    out['A'] = {'probes': a, 'commonality_r2': pid}

    # B. latent-z JEPA view
    b = {
        't->s_v': probe(xt, sv), 't+Y->s_v': probe(np.hstack([xt, ys]), sv), 't+Yperm->s_v': probe(np.hstack([xt, y_perm]), sv),
        'v->s_t': probe(xv, st_), 'v+Y->s_t': probe(np.hstack([xv, ys]), st_), 'v+Yperm->s_t': probe(np.hstack([xv, y_perm]), st_),
        'Y->s_v': probe(ys, sv), 'Y->s_t': probe(ys, st_),
    }
    gains = {}
    for g in GROUPS:
        gains[g] = {
            'gain_v (U_v+S)': b['t+Y->s_v'][g]['info'] - b['t->s_v'][g]['info'],
            'gain_v control': b['t+Yperm->s_v'][g]['info'] - b['t->s_v'][g]['info'],
            'gain_t (U_t+S)': b['v+Y->s_t'][g]['info'] - b['v->s_t'][g]['info'],
            'gain_t control': b['v+Yperm->s_t'][g]['info'] - b['v->s_t'][g]['info'],
            'gain_v r2': b['t+Y->s_v'][g]['r2'] - b['t->s_v'][g]['r2'],
            'gain_t r2': b['v+Y->s_t'][g]['r2'] - b['v->s_t'][g]['r2'],
        }
    out['B'] = {'probes': b, 'gains': gains}

    # C. popularity content of Y and of the modalities
    out['C'] = {'Y->logdeg': probe(ys, logdeg), 'v->logdeg': probe(xv, logdeg), 't->logdeg': probe(xt, logdeg)}
    return out


def debias(y, degree):
    """Remove the part of Y linearly explained by [log deg, log deg^2]."""
    ld = np.log1p(degree.astype(np.float64))
    x = np.column_stack([np.ones_like(ld), ld, ld ** 2])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ coef


# ------------------------------------------------------------------ printing
def fmt_probe(name, res):
    return '  {:<14}'.format(name) + '  '.join(
        '{}: I={:7.2f} R2={:6.3f}'.format(g, res[g]['info'], res[g]['r2']) for g in GROUPS) + '  (alpha={:g})'.format(res['alpha'])


def report(out):
    print('\n' + '=' * 110)
    print('[{}] items with Y: {} | groups {}'.format(out['tag'], out['n_items'], out['group_sizes']))
    print('A. Predicting Y (collaborative item embedding) from content   [held-out, nats summed over Y dims]')
    for k, r in out['A']['probes'].items():
        print(fmt_probe(k, r))
    print('  share of Y variance (commonality):')
    for g in GROUPS:
        c = out['A']['commonality_r2'][g]
        print('    {:<5} R={:+.3f}  U_v={:+.3f}  U_t={:+.3f}  unexplained={:.3f}'.format(
            g, c['R'], c['U_v'], c['U_t'], c['unexplained']))
    print('B. Latent-z JEPA: does Y help one modality predict the other?   [target = other modality, PCA dims]')
    for k, r in out['B']['probes'].items():
        print(fmt_probe(k, r))
    for g in GROUPS:
        gg = out['B']['gains'][g]
        print('  {:<5} gain_v={:+.2f} nats / {:+.3f} R2 (perm {:+.2f})   gain_t={:+.2f} nats / {:+.3f} R2 (perm {:+.2f})'.format(
            g, gg['gain_v (U_v+S)'], gg['gain_v r2'], gg['gain_v control'],
            gg['gain_t (U_t+S)'], gg['gain_t r2'], gg['gain_t control']))
    print('C. Popularity: R2 of log(1+degree) from Y / v / t: ' + '  '.join(
        '{}={:.3f}'.format(k.split('->')[0], r['all']['r2']) for k, r in out['C'].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='baby')
    ap.add_argument('--v_file', default=None, help='default: vision_feature_file of the dataset yaml')
    ap.add_argument('--t_file', default=None, help='default: text_feature_file of the dataset yaml')
    ap.add_argument('--y_file', default='cf_item_emb.npy')
    ap.add_argument('--pca', type=int, default=128, help='input dims per modality (PCA)')
    ap.add_argument('--target_dim', type=int, default=64, help='dims of s_v / s_t targets (PCA)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--alphas', type=float, nargs='+', default=[1, 10, 100, 1000, 10000])
    ap.add_argument('--min_degree', type=int, default=1, help='only items with >= this many train interactions')
    ap.add_argument('--groups', type=float, nargs=2, default=[0.2, 0.5], help='head / mid cut-offs (item share)')
    ap.add_argument('--debias', action='store_true', help='also run with popularity-debiased Y')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    v, t, y, degree, n_items = load_data(args)
    group = popularity_groups(degree, args.groups)   # groups over ALL items, as in the evaluator
    print('n_items={} v{} t{} Y{} | items with degree>={}: {}'.format(
        n_items, v.shape, t.shape, y.shape, args.min_degree, int((degree >= args.min_degree).sum())))

    results = [run_all(v, t, y, degree, group, args, 'Y')]
    if args.debias:
        results.append(run_all(v, t, debias(y, degree), degree, group, args, 'Y_debiased'))
    for r in results:
        report(r)

    os.makedirs('log', exist_ok=True)
    path = os.path.join('log', 'probe_latent_z_{}{}.json'.format(args.dataset, ('_' + args.tag) if args.tag else ''))
    with open(path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=1)
    print('\nSaved', path)


if __name__ == '__main__':
    main()
