# coding: utf-8
"""
Pretrain an ID-only LightGCN to obtain the collaborative target Y
for Predictive PID (fuse='vpid').

Y must NOT contain any visual/text information, otherwise the
decomposition is circular. This script therefore uses only user/item
IDs and TRAINING interactions (x_label == 0), with the same split and
ID space as the main framework (utils.dataset.RecDataset).

Run from the repo root (same as main.py):
    python src/pretrain_cf.py -d baby

Outputs (in data/<dataset>/):
    cf_item_emb.npy     [n_items, dim]  -> target Y
    cf_user_emb.npy     [n_users, dim]
    cf_item_degree.npy  [n_items]       -> #train interactions per item
                                           (degree 0 = no reliable Y)
"""
import os
import argparse
from time import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.utils import init_seed


def build_graph(train_df, uid_field, iid_field, n_users, n_items, device):
    rows = train_df[uid_field].values.astype(np.int64)
    cols = train_df[iid_field].values.astype(np.int64)
    R = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    ).tocsr()
    R.data[:] = 1.0  # remove duplicate counts

    A = sp.bmat([[None, R], [R.T, None]], format='coo')
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D = sp.diags(d_inv_sqrt)
    L = (D @ A @ D).tocoo()

    idx = torch.from_numpy(np.vstack([L.row, L.col]).astype(np.int64))
    val = torch.from_numpy(L.data.astype(np.float32))
    adj = torch.sparse_coo_tensor(idx, val, L.shape).coalesce().to(device)
    return adj, R


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers, adj):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.adj = adj
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        outs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            outs.append(x)
        x = torch.stack(outs, dim=0).mean(dim=0)
        return x[:self.n_users], x[self.n_users:]

    def bpr_loss(self, u, i, j, reg):
        users, items = self.propagate()
        u_e, i_e, j_e = users[u], items[i], items[j]
        pos = (u_e * i_e).sum(-1)
        neg = (u_e * j_e).sum(-1)
        loss = F.softplus(-(pos - neg)).mean()
        ego = (self.user_emb(u).pow(2).sum() + self.item_emb(i).pow(2).sum()
               + self.item_emb(j).pow(2).sum()) / u.size(0)
        return loss + reg * ego


def sample_negatives(users, R, n_items, rng, n_tries=5):
    neg = rng.integers(0, n_items, size=len(users))
    for _ in range(n_tries):
        hit = np.asarray(R[users, neg]).flatten() > 0
        if not hit.any():
            break
        neg[hit] = rng.integers(0, n_items, size=int(hit.sum()))
    return neg


@torch.no_grad()
def evaluate(users_emb, items_emb, eval_df, uid_field, iid_field, R_train, k=20, batch=4096):
    gt = eval_df.groupby(uid_field)[iid_field].apply(set)
    eval_users = gt.index.values.astype(np.int64)
    recalls, ndcgs = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for s in range(0, len(eval_users), batch):
        ub = eval_users[s:s + batch]
        scores = users_emb[torch.from_numpy(ub).to(users_emb.device)] @ items_emb.t()
        sub = R_train[ub].tocoo()
        if sub.nnz > 0:
            scores[torch.from_numpy(sub.row.astype(np.int64)).to(scores.device),
                   torch.from_numpy(sub.col.astype(np.int64)).to(scores.device)] = -1e10
        topk = torch.topk(scores, k, dim=-1).indices.cpu().numpy()
        for row, u in enumerate(ub):
            g = gt[u]
            hits = np.array([1.0 if it in g else 0.0 for it in topk[row]])
            recalls.append(hits.sum() / len(g))
            idcg = discounts[:min(len(g), k)].sum()
            ndcgs.append((hits * discounts).sum() / idcg)
    return float(np.mean(recalls)), float(np.mean(ndcgs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10, help='in #evaluations')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--out_prefix', type=str, default='cf')
    args, _ = parser.parse_known_args()

    config = Config('LightGCN', args.dataset, {'gpu_id': args.gpu_id})
    device = config['device']
    init_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = RecDataset(config)
    n_users, n_items = dataset.get_user_num(), dataset.get_item_num()
    train_ds, valid_ds, test_ds = dataset.split()
    uid, iid = config['USER_ID_FIELD'], config['ITEM_ID_FIELD']
    train_df, valid_df, test_df = train_ds.df, valid_ds.df, test_ds.df
    print(f'[CF] users={n_users} items={n_items} train={len(train_df)} '
          f'valid={len(valid_df)} test={len(test_df)} device={device}')

    adj, R = build_graph(train_df, uid, iid, n_users, n_items, device)
    model = LightGCN(n_users, n_items, args.dim, args.layers, adj).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    pairs_u = train_df[uid].values.astype(np.int64)
    pairs_i = train_df[iid].values.astype(np.int64)

    best, best_epoch, bad = -1.0, -1, 0
    best_u = best_i = None
    for epoch in range(args.epochs):
        t0 = time()
        perm = rng.permutation(len(pairs_u))
        total = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            u, i = pairs_u[b], pairs_i[b]
            j = sample_negatives(u, R, n_items, rng)
            loss = model.bpr_loss(
                torch.from_numpy(u).to(device),
                torch.from_numpy(i).to(device),
                torch.from_numpy(j).to(device),
                args.reg,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        if (epoch + 1) % args.eval_every == 0:
            with torch.no_grad():
                ue, ie = model.propagate()
            rec, ndcg = evaluate(ue, ie, valid_df, uid, iid, R)
            flag = ''
            if rec > best:
                best, best_epoch, bad = rec, epoch, 0
                best_u, best_i = ue.cpu().numpy(), ie.cpu().numpy()
                flag = ' *'
            else:
                bad += 1
            print(f'[CF] epoch {epoch} loss={total:.4f} valid R@20={rec:.4f} '
                  f'N@20={ndcg:.4f} ({time() - t0:.1f}s){flag}')
            if bad >= args.patience:
                print(f'[CF] early stop, best epoch {best_epoch}')
                break

    # test with best embeddings (reference number for the CF teacher)
    rec, ndcg = evaluate(torch.from_numpy(best_u).to(device), torch.from_numpy(best_i).to(device),
                         test_df, uid, iid, R)
    print(f'[CF] best epoch {best_epoch}: valid R@20={best:.4f} | test R@20={rec:.4f} N@20={ndcg:.4f}')

    degree = np.asarray(R.sum(axis=0)).flatten().astype(np.int64)
    out_dir = dataset.dataset_path
    np.save(os.path.join(out_dir, f'{args.out_prefix}_item_emb.npy'), best_i.astype(np.float32))
    np.save(os.path.join(out_dir, f'{args.out_prefix}_user_emb.npy'), best_u.astype(np.float32))
    np.save(os.path.join(out_dir, f'{args.out_prefix}_item_degree.npy'), degree)
    print(f'[CF] saved to {out_dir}: {args.out_prefix}_item_emb.npy {best_i.shape}, '
          f'items with degree 0: {(degree == 0).sum()}')


if __name__ == '__main__':
    main()
