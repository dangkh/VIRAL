# coding: utf-8

"""
################################
"""
import os
import numpy as np
import pandas as pd
import torch
from utils.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_local_time


# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}


class TopKEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """

    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        self._check_args()
        # popularity-group evaluation (head / mid / tail items by TRAIN interaction count)
        self.pop_eval = True if config['pop_eval'] is None else bool(config['pop_eval'])
        self.pop_topk = [k for k in (config['pop_eval_topk'] or [10, 20]) if k <= max(self.topk)]
        self.pop_groups = list(config['pop_groups'] or [0.2, 0.5])   # head = top 20% items, mid = up to 50%
        self._pop_cache_key = None
        self._item_group = None

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}-{}-idx{}-top{}-{}.csv'.format(
                model_name, dataset_name, idx, max_k, get_local_time()))
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id']+['top_'+str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)
        assert len(pos_len_list) == len(topk_index)
        # if recom right?
        bool_rec_matrix = []
        for m, n in zip(pos_items, topk_index):
            bool_rec_matrix.append([True if i in m else False for i in n])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)
        if self.pop_eval and self.pop_topk:
            metric_dict.update(self._popularity_metrics(eval_data, pos_items, topk_index, bool_rec_matrix))
        return metric_dict

    # ------------------------------------------------------------------
    # Popularity-aware evaluation
    # ------------------------------------------------------------------
    GROUP_NAMES = ('head', 'mid', 'tail')

    def _get_item_groups(self, eval_data):
        """0 = head, 1 = mid, 2 = tail, ranked by number of TRAIN interactions (ties: lower id first)."""
        train = eval_data.additional_dataset
        key = (id(train), len(train.df))
        if key != self._pop_cache_key:
            n_items = int(eval_data.dataset.item_num)
            degree = np.bincount(train.df[train.iid_field].values.astype(np.int64), minlength=n_items)
            order = np.argsort(-degree, kind='stable')
            cut_head = int(np.ceil(self.pop_groups[0] * n_items))
            cut_mid = int(np.ceil(self.pop_groups[1] * n_items))
            group = np.full(n_items, 2, dtype=np.int64)
            group[order[:cut_mid]] = 1
            group[order[:cut_head]] = 0
            self._item_group, self._pop_cache_key = group, key
        return self._item_group

    def _popularity_metrics(self, eval_data, pos_items, topk_index, bool_rec_matrix):
        group = self._get_item_groups(eval_data)
        n_items = len(group)
        n_users = len(pos_items)
        # number of test positives per user and group
        pos_cnt = np.zeros((n_users, 3), dtype=np.int64)
        for u, items in enumerate(pos_items):
            np.add.at(pos_cnt[u], group[np.asarray(items, dtype=np.int64)], 1)
        top_group = group[topk_index]                       # users x K
        max_k = max(self.pop_topk)
        discount = 1.0 / np.log2(np.arange(2, max_k + 2))
        idcg_table = np.concatenate([[0.0], np.cumsum(discount)])
        out = {}
        for k in self.pop_topk:
            hits = bool_rec_matrix[:, :k]
            tg = top_group[:, :k]
            for g, name in enumerate(self.GROUP_NAMES):
                valid = pos_cnt[:, g] > 0
                if not valid.any():
                    continue
                hg = hits & (tg == g)
                recall = hg.sum(1)[valid] / pos_cnt[valid, g]
                dcg = (hg * discount[:k]).sum(1)[valid]
                idcg = idcg_table[np.minimum(pos_cnt[valid, g], k)]
                out['recall_{}@{}'.format(name, k)] = round(float(recall.mean()), 4)
                out['ndcg_{}@{}'.format(name, k)] = round(float((dcg / idcg).mean()), 4)
            rec = topk_index[:, :k]
            exposure = np.bincount(rec.ravel(), minlength=n_items).astype(np.float64)
            x = np.sort(exposure)
            i = np.arange(1, n_items + 1)
            gini = float(((2 * i - n_items - 1) * x).sum() / (n_items * x.sum())) if x.sum() > 0 else 0.0
            out['coverage@{}'.format(k)] = round(float((exposure > 0).mean()), 4)
            out['aplt@{}'.format(k)] = round(float((tg == 2).mean()), 4)       # share of tail items in top-k
            out['gini@{}'.format(k)] = round(gini, 4)
        return out

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users
        Returns:
            np.ndarray: a matrix which contains the metrics result
        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [topk_metrics[metric.lower()] for metric in self.metrics]) \
               + '], TopK:[' + ', '.join(map(str, self.topk)) + ']'
        return mesg
