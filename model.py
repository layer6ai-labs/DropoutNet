import tensorflow as tf


def dense_batch_fc_tanh(x, units, phase, scope, do_norm=False):
    """
    convenience function to build tanh blocks in DeepCF
    tanh is found to work better for DeepCF nets
    constitutes of: FC -> batch norm -> tanh activation

    x: input
    units: # of hidden units in FC
    phase: boolean flag whether we are training, required by batch norm
    scope: name of block
    do_norm: boolean flag to do batch norm after FC
    """

    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init)
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer())
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.contrib.layers.batch_norm(
                h1,
                decay=0.9,
                center=True,
                scale=True,
                is_training=phase,
                scope=scope + '_bn')
            return tf.nn.tanh(h2, scope + '_tanh')
        else:
            return tf.nn.tanh(h1, scope + '_tanh')


class DeepCF:
    """
    main model class implementing DeepCF
    also stores states for fast candidate generation

    latent_rank_in: rank of preference model input
    user_content_rank: rank of user content input
    item_content_rank: rank of item content input
    model_select: array of number of hidden unit,
        i.e. [200,100] indicate two hidden layer with 200 units followed by 100 units
    rank_out: rank of latent model output

    """

    def __init__(self, latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):

        self.rank_in = latent_rank_in
        self.phi_u_dim = user_content_rank
        self.phi_v_dim = item_content_rank
        self.model_select = model_select
        self.rank_out = rank_out

        # inputs
        self.Uin = None
        self.Vin = None
        self.Ucontent = None
        self.Vcontent = None
        self.phase = None
        self.target = None
        self.eval_trainR = None
        self.U_pref_tf = None
        self.V_pref_tf = None
        self.rand_target_ui = None

        # outputs in the model

        self.preds = None
        self.updates = None
        self.loss = None

        self.U_embedding = None
        self.V_embedding = None

        self.lr_placeholder = None

        # predictor
        self.tf_topk_vals = None
        self.tf_topk_inds = None
        self.preds_random = None
        self.tf_latent_topk_cold = None
        self.tf_latent_topk_warm = None
        self.eval_preds_warm = None
        self.eval_preds_cold = None

    def build_model(self):
        """
        set up tf components for main DeepCF net
        call after setting up desired tf state (cpu/gpu etc...)

        Note: should use GPU
        """
        self.lr_placeholder = tf.placeholder(tf.float32, shape=[], name='learn_rate')
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.target = tf.placeholder(tf.float32, shape=[None], name='target')

        self.Uin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='U_in_raw')
        self.Vin = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='V_in_raw')
        if self.phi_u_dim>0:
            self.Ucontent = tf.placeholder(tf.float32, shape=[None, self.phi_u_dim], name='U_content')
            u_concat = tf.concat([self.Uin, self.Ucontent], 1)
        else:
            u_concat = self.Uin

        if self.phi_v_dim>0:
            self.Vcontent = tf.placeholder(tf.float32, shape=[None, self.phi_v_dim], name='V_content')
            v_concat = tf.concat([self.Vin, self.Vcontent], 1)
        else:
            v_concat = self.Vin

        print ('\tu_concat.shape=%s' % str(u_concat.get_shape()))
        print ('\tv_concat.shape=%s' % str(v_concat.get_shape()))

        u_last = u_concat
        v_last = v_concat
        for ihid, hid in enumerate(self.model_select):
            u_last = dense_batch_fc_tanh(u_last, hid, self.phase, 'user_layer_%d' % (ihid + 1), do_norm=True)
            v_last = dense_batch_fc_tanh(v_last, hid, self.phase, 'item_layer_%d' % (ihid + 1), do_norm=True)

        with tf.variable_scope("self.U_embedding"):
            u_emb_w = tf.Variable(tf.truncated_normal([u_last.get_shape().as_list()[1], self.rank_out], stddev=0.01),
                                  name='u_emb_w')
            u_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='u_emb_b')
            self.U_embedding = tf.matmul(u_last, u_emb_w) + u_emb_b

        with tf.variable_scope("V_embedding"):
            v_emb_w = tf.Variable(tf.truncated_normal([v_last.get_shape().as_list()[1], self.rank_out], stddev=0.01),
                                  name='v_emb_w')
            v_emb_b = tf.Variable(tf.zeros([1, self.rank_out]), name='v_emb_b')
            self.V_embedding = tf.matmul(v_last, v_emb_w) + v_emb_b

        with tf.variable_scope("loss"):
            preds = tf.multiply(self.U_embedding, self.V_embedding)
            self.preds = tf.reduce_sum(preds, 1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.preds, self.target))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.updates = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9).minimize(self.loss)

    def build_predictor(self, recall_at, num_candidates):
        """
        set up tf components for prediction and target selection
        call after setting up desired tf state (cpu/gpu etc...)

        Note: should use CPU, as large inputs are expected

        :param recall_at: truncation to compute recall
        :param num_candidates: number of candidates
        :return:
        """
        self.eval_trainR = tf.sparse_placeholder(
            dtype=tf.float32, shape=[None, None], name='trainR_sparse_CPU')

        with tf.variable_scope("eval"):
            embedding_prod_cold = tf.matmul(self.U_embedding, self.V_embedding, transpose_b=True, name='pred_all_items')
            embedding_prod_warm = tf.sparse_add(embedding_prod_cold, self.eval_trainR)
            _, self.eval_preds_cold = tf.nn.top_k(embedding_prod_cold, k=recall_at[-1], sorted=True,
                                                  name='topK_net_cold')
            _, self.eval_preds_warm = tf.nn.top_k(embedding_prod_warm, k=recall_at[-1], sorted=True,
                                                  name='topK_net_warm')
        with tf.variable_scope("select_targets"):
            self.U_pref_tf = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='u_pref')
            self.V_pref_tf = tf.placeholder(tf.float32, shape=[None, self.rank_in], name='v_pref')
            self.rand_target_ui = tf.placeholder(tf.int32, shape=[None, None], name='rand_target_ui')
            preds_pref = tf.matmul(self.U_pref_tf, self.V_pref_tf, transpose_b=True)
            tf_topk_vals, tf_topk_inds = tf.nn.top_k(preds_pref, k=num_candidates, sorted=True, name='top_targets')
            self.tf_topk_vals = tf.reshape(tf_topk_vals, [-1], name='select_y_vals')
            self.tf_topk_inds = tf.reshape(tf_topk_inds, [-1], name='select_y_inds')
            preds_random = tf.gather_nd(preds_pref, self.rand_target_ui)
            self.preds_random = tf.reshape(preds_random, [-1], name='random_y_inds')

        # tf matmul-topk to get eval on latent
        with tf.variable_scope("latent_eval"):
            preds_pref_latent_warm = tf.sparse_add(preds_pref, self.eval_trainR)
            _, self.tf_latent_topk_cold = tf.nn.top_k(preds_pref, k=recall_at[-1], sorted=True, name='topK_latent_cold')
            _, self.tf_latent_topk_warm = tf.nn.top_k(preds_pref_latent_warm, k=recall_at[-1], sorted=True,
                                                      name='topK_latent_warm')

    def get_eval_dict(self, _i, _eval_start, _eval_finish, eval_data):
        """
        packaging method to iterate evaluation data, select from start:finish
        should be passed directly to batch method

        :param _i: slice id
        :param _eval_start: integer beginning of slice
        :param _eval_finish: integer end of slice
        :param eval_data: package EvalData obj
        :return:
        """
        _eval_dict = {
            self.Uin: eval_data.U_pref_test[_eval_start:_eval_finish, :],
            self.Vin: eval_data.V_pref_test,
            self.Vcontent: eval_data.V_content_test,
            self.phase: 0
        }
        if self.Ucontent!=None: 
            _eval_dict[self.Ucontent]= eval_data.U_content_test[_eval_start:_eval_finish, :]
        if not eval_data.is_cold:
            _eval_dict[self.eval_trainR] = eval_data.tf_eval_train[_i]
        return _eval_dict

    def get_eval_dict_latent(self, _i, _eval_start, _eval_finish, eval_data, u_pref, v_pref):
        """
        packaging method to iterate evaluation data, select from start:finish
        uses preference input
        should be passed directly to batch method

        :param _i: slice id
        :param _eval_start: integer beginning of slice
        :param _eval_finish: integer end of slice
        :param eval_data: package EvalData obj
        :param u_pref: user latent input to slice
        :param v_pref: item latent input to slice
        :return:
        """
        _eval_dict = {
            self.U_pref_tf: u_pref[eval_data.test_user_ids[_eval_start:_eval_finish], :],
            self.V_pref_tf: v_pref[eval_data.test_item_ids, :]
        }
        if not eval_data.is_cold:
            _eval_dict[self.eval_trainR] = eval_data.tf_eval_train[_i]
        return _eval_dict
