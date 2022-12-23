import numpy as np
import tensorflow as tf
import scipy.sparse
import utils
import pandas as pd

"""
This module contains class and methods related to data used in DropoutNet  
"""


def load_eval_data(test_file, test_id_file, name, cold, train_data, citeu=False):
    timer = utils.timer()
    with open(test_id_file) as f:
        test_item_ids = [int(line) for line in f]
        test_data = pd.read_csv(test_file, delimiter=",", header=-1, dtype=np.int32).values.ravel()
        if citeu:
            test_data = test_data.view(
            dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32)])
        else:
            test_data = test_data.view(
            dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32), ('date', np.int32)])
        timer.toc('read %s triplets %s' % (name, test_data.shape)).tic()
        eval_data = EvalData(
            test_data,
            test_item_ids,
            is_cold=cold,
            train=train_data
        )
        timer.toc('loaded %s' % name).tic()
        print(eval_data.get_stats_string())
        return eval_data


class EvalData:
    """
    EvalData:
        EvalData packages test triplet (user, item, score) into appropriate formats for evaluation

        Compact Indices:
            Specifically, this builds compact indices and stores mapping between original and compact indices.
            Compact indices only contains:
                1) items in test set
                2) users who interacted with such test items
            These compact indices speed up testing significantly by ignoring irrelevant users or items

        Args:
            test_triplets(int triplets): user-item-interaction_value triplet to build the test data
            train(int triplets): user-item-interaction_value triplet from train data

        Attributes:
            is_cold(boolean): whether test data is used for cold start problem
            test_item_ids(list of int): maps compressed item ids to original item ids (via position)
            test_item_ids_map(dictionary of int->int): maps original item ids to compressed item ids
            test_user_ids(list of int): maps compressed user ids to original user ids (via position)
            test_user_ids_map(dictionary of int->int): maps original user ids to compressed user ids
            R_test_inf(scipy lil matrix): pre-built compressed test matrix
            R_train_inf(scipy lil matrix): pre-built compressed train matrix for testing

            other relevant input/output exposed from tensorflow graph

    """

    def __init__(self, test_triplets, test_item_ids, is_cold, train):
        # build map both-ways between compact and original indices
        # compact indices only contains:
        #  1) items in test set
        #  2) users who interacted with such test items

        self.is_cold = is_cold

        self.test_item_ids = test_item_ids
        # test_item_ids_map
        self.test_item_ids_map = {iid: i for i, iid in enumerate(self.test_item_ids)}

        _test_ij_for_inf = [(t[0], t[1]) for t in test_triplets if t[1] in self.test_item_ids_map]
        # test_user_ids
        self.test_user_ids = np.unique(test_triplets['uid'])
        # test_user_ids_map
        self.test_user_ids_map = {user_id: i for i, user_id in enumerate(self.test_user_ids)}

        _test_i_for_inf = [self.test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
        _test_j_for_inf = [self.test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
        self.R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(_test_i_for_inf)),
             (_test_i_for_inf, _test_j_for_inf)),
            shape=[len(self.test_user_ids), len(self.test_item_ids)]
        ).tolil(copy=False)

        train_ij_for_inf = [(self.test_user_ids_map[_t[0]], self.test_item_ids_map[_t[1]]) for _t
                            in train
                            if _t[1] in self.test_item_ids_map and _t[0] in self.test_user_ids_map]
        if self.is_cold and len(train_ij_for_inf) is not 0:
            raise Exception('using cold dataset, but data is not cold!')
        if not self.is_cold and len(train_ij_for_inf) is 0:
            raise Exception('using warm datset, but data is not warm!')

        self.R_train_inf = None if self.is_cold else scipy.sparse.coo_matrix((
            np.ones(len(train_ij_for_inf)),
            zip(*train_ij_for_inf)), shape=self.R_test_inf.shape).tolil(copy=False)

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
        self.V_content_test = item_content[self.test_item_ids, :]
        if scipy.sparse.issparse(self.V_content_test):
            self.V_content_test = self.V_content_test.todense()
        if user_content!=None:
            self.U_content_test = user_content[self.test_user_ids, :]
            if scipy.sparse.issparse(self.U_content_test):
                self.U_content_test = self.U_content_test.todense()
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
