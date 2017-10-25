import numpy as np
import tensorflow as tf
import scipy.sparse


class EvalData:
    def __init__(self, _test_triplets, _test_item_ids, is_cold, train):
        # build map both-ways between compact and original indices
        # compact indices only contains:
        #  1) items in test set
        #  2) users who interacted with such test items

        self.is_cold = is_cold

        self.test_item_ids = _test_item_ids
        # test_item_ids_map
        self.test_item_ids_map = {_id: _i for _i, _id in enumerate(self.test_item_ids)}

        _test_ij_for_inf = [(_t[0], _t[1]) for _t in _test_triplets if _t[1] in self.test_item_ids_map]
        # test_user_ids
        self.test_user_ids = np.unique(_test_triplets['uid'])
        # test_user_ids_map
        self.test_user_ids_map = {_id: _i for _i, _id in enumerate(self.test_user_ids)}

        _test_i_for_inf = [self.test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
        _test_j_for_inf = [self.test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
        self.R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)),
             (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(self.test_user_ids), len(self.test_item_ids)]
        ).tolil(copy=False)

        _train_ij_for_inf = [(self.test_user_ids_map[_t[0]], self.test_item_ids_map[_t[1]]) for _t
                             in train
                             if _t[1] in self.test_item_ids_map and _t[0] in self.test_user_ids_map]
        if self.is_cold and len(_train_ij_for_inf) is not 0:
            raise Exception('using cold dataset, but data is not cold!')
        if not self.is_cold and len(_train_ij_for_inf) is 0:
            raise Exception('using warm datset, but data is not warm!')

        self.R_train_inf = None if self.is_cold else scipy.sparse.coo_matrix((
            np.ones(len(_train_ij_for_inf)),
            zip(*_train_ij_for_inf)), shape=self.R_test_inf.shape).tolil(copy=False)

        # allocate fields
        self.U_pref_test = None
        self.V_pref_test = None
        self.V_content_test = None
        self.U_content_test = None
        self.tf_eval_train = None
        self.tf_eval_test = None
        self.eval_batch = None

    def init_tf(self, user_factors, item_factors, user_content, item_content, eval_run_batchsize):
        self.U_pref_test = user_factors[self.test_user_ids, :]
        self.V_pref_test = item_factors[self.test_item_ids, :]
        self.V_content_test = item_content[self.test_item_ids, :].todense()
        self.U_content_test = user_content[self.test_user_ids, :].todense()
        eval_l = self.R_test_inf.shape[0]
        self.eval_batch = [(x, min(x + eval_run_batchsize, eval_l)) for x
                           in xrange(0, eval_l, eval_run_batchsize)]

        self.tf_eval_train = []
        self.tf_eval_test = []

        if not self.is_cold:
            for (eval_start, eval_finish) in self.eval_batch:
                _ui = self.R_train_inf[eval_start:eval_finish, :].tocoo()
                _ui = zip(_ui.row, _ui.col)
                self.tf_eval_train.append(
                    tf.SparseTensorValue(
                        indices=_ui,
                        values=np.full(len(_ui), -100000, dtype=np.float32),
                        dense_shape=[eval_finish - eval_start, self.R_train_inf.shape[1]]
                    )
                )

    def get_stats_string(self):
        return ('\tn_test_users:[%d]\n\tn_test_items:[%d]' % (len(self.test_user_ids), len(self.test_item_ids))
                + '\n\tR_train_inf: %s' % (
                    'no R_train_inf for cold' if self.is_cold else 'shape=%s nnz=[%d]' % (
                        str(self.R_train_inf.shape), len(self.R_train_inf.nonzero()[0])
                    )
                )
                + '\n\tR_test_inf: shape=%s nnz=[%d]' % (
                    str(self.R_test_inf.shape), len(self.R_test_inf.nonzero()[0])
                ))
