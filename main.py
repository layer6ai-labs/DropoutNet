import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn import datasets
from data import EvalData
import model


def load_eval_data(test_file, test_id_file, name, cold, train_data):
    with open(test_id_file) as f:
        test_item_ids = [int(line) for line in f]
        test_data = pd.read_csv(test_file, delimiter=",", header=-1, dtype=np.int32).values.ravel().view(
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


timer = utils.timer(name='main').tic()

# SETTINGS

DATA_PATH = '/home/guang/Data/recsys2017/eval'
# CHECKPOINT_PATH = '/datadrive/tmp/tf_save/'
# TB_LOG_PATH = '/datadrive/tmp/citeu.tf.log/'
CHECKPOINT_PATH = None
TB_LOG_PATH = None
model_select = [800, 400]

# DEFINITIONS

N_USERS = 1497020 + 1
N_ITEMS = 1306054 + 1
SPLIT_FOLDER = DATA_PATH + '/warm'

U_FILE = DATA_PATH + '/trained/warm/U.csv.bin'
V_FILE = DATA_PATH + '/trained/warm/V.csv.bin'
USER_CONTENT_FILE = DATA_PATH + '/user_features_0based.txt'
ITEM_CONTENT_FILE = DATA_PATH + '/item_features_0based.txt'
TRAIN_FILE = SPLIT_FOLDER + '/train.csv'
TEST_WARM_FILE = SPLIT_FOLDER + '/test_warm.csv'
TEST_WARM_IID_FILE = SPLIT_FOLDER + '/test_warm_item_ids.csv'
TEST_COLD_USER_FILE = SPLIT_FOLDER + '/test_cold_user.csv'
TEST_COLD_USER_IID_FILE = SPLIT_FOLDER + '/test_cold_user_item_ids.csv'
TEST_COLD_ITEM_FILE = SPLIT_FOLDER + '/test_cold_item.csv'
TEST_COLD_ITEM_IID_FILE = SPLIT_FOLDER + '/test_cold_item_item_ids.csv'

_nusers = N_USERS
_nitems = N_ITEMS
_rank_out = 200
_ubatch_size = 1000
_nscores_user = 5000
_dbatch_size = 1000
_deepcf_dropout = 0.5
_recall_at = range(50, 550, 50)
_eval_batch_size = 1000
_max_data_per_step = 2500000
_eval_every = 2
_num_epoch = 10

_lr = 0.005
_mom = 0.9
_decay_lr_every = 50
_lr_decay = 0.1

experiment = '%s_%s_2stage' % (
    datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
    '-'.join(str(x / 100) for x in model_select) if model_select else 'simple'
)
print('running: ' + experiment)
_tf_ckpt_file = None if CHECKPOINT_PATH is None else CHECKPOINT_PATH + experiment + '/tf_checkpoint'

# END DEFINITIONS

# load preference data
timer.tic()
U_pref = np.fromfile(U_FILE, dtype=np.float32).reshape(N_USERS, 200)
V_pref = np.fromfile(V_FILE, dtype=np.float32).reshape(N_ITEMS, 200)

timer.toc('loaded U:%s,V:%s' % (str(U_pref.shape), str(V_pref.shape))).tic()

U_pref.tofile(U_FILE + '.bin')
V_pref.tofile(V_FILE + '.bin')

# preprocessing
scaler_U_pref, U_pref_scaled = utils.prep_standardize(U_pref)
scaler_V_pref, V_pref_scaled = utils.prep_standardize(V_pref)
timer.toc('standardized U,V').tic()

# load content data
timer.tic()
U_content, _ = datasets.load_svmlight_file(USER_CONTENT_FILE, zero_based=True, dtype=np.float32)
U_content = U_content.tolil(copy=False)
timer.toc('loaded user feature sparse matrix: %s' % (str(U_content.shape))).tic()
V_content, _ = datasets.load_svmlight_file(ITEM_CONTENT_FILE, zero_based=True, dtype=np.float32)
V_content = V_content.tolil(copy=False)
timer.toc('loaded item feature sparse matrix: %s' % (str(V_content.shape))).tic()

# load split
timer.tic()
train = pd.read_csv(TRAIN_FILE, delimiter=",", header=-1, dtype=np.int32).values.ravel().view(
    dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32), ('date', np.int32)])
user_indices = np.unique(train['uid'])
item_indices = np.unique(train['iid'])
timer.toc('read train triplets %s' % train.shape).tic()

eval_warm = load_eval_data(TEST_WARM_FILE, TEST_WARM_IID_FILE, name='eval_warm', cold=False, train_data=train)
eval_cold_user = load_eval_data(TEST_COLD_USER_FILE, TEST_COLD_USER_IID_FILE, name='eval_cold_user', cold=True,
                                train_data=train)
eval_cold_item = load_eval_data(TEST_COLD_ITEM_FILE, TEST_COLD_ITEM_IID_FILE, name='eval_cold_item', cold=True,
                                train_data=train)

print('running: ' + experiment)

timer = utils.timer(name='main').tic()

# append pref factors for faster dropout
V_pref_expanded = np.vstack([V_pref_scaled, np.zeros_like(V_pref_scaled[0, :])])
V_pref_last = V_pref_scaled.shape[0]
U_pref_expanded = np.vstack([U_pref_scaled, np.zeros_like(U_pref_scaled[0, :])])
U_pref_last = U_pref_scaled.shape[0]
timer.toc('initialized numpy data for tf')

# prep eval
eval_batch_size = _eval_batch_size
timer.tic()
eval_warm.init_tf(U_pref_scaled, V_pref_scaled, U_content, V_content, eval_batch_size)
timer.toc('initialized eval_warm for tf').tic()
eval_cold_user.init_tf(U_pref_scaled, V_pref_scaled, U_content, V_content, eval_batch_size)
timer.toc('initialized eval_cold_user for tf').tic()
eval_cold_item.init_tf(U_pref_scaled, V_pref_scaled, U_content, V_content, eval_batch_size)
timer.toc('initialized eval_cold_item for tf').tic()

_rank_in = U_pref.shape[1]
_phi_u_dim = U_content.shape[1]
_phi_v_dim = V_content.shape[1]

deepcf = model.DeepCF(latent_rank_in=U_pref.shape[1],
                      user_content_rank=U_content.shape[1],
                      item_content_rank=V_content.shape[1],
                      model_select=model_select,
                      rank_out=_rank_out)

config = tf.ConfigProto(allow_soft_placement=True)

with tf.device('/gpu:0'):
    deepcf.build_model()

with tf.device('/cpu:0'):
    deepcf.build_predictor(_recall_at, _nscores_user)

with tf.Session(config=config) as sess:
    tf_saver = None if _tf_ckpt_file is None else tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = None if TB_LOG_PATH is None else tf.summary.FileWriter(
        TB_LOG_PATH + experiment, sess.graph)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    timer.toc('initialized tf')

    row_index = np.copy(user_indices)
    n_step = 0
    best_cold_user = 0
    best_cold_item = 0
    best_warm = 0
    n_batch_trained = 0
    best_step = 0
    for epoch in range(_num_epoch):
        np.random.shuffle(row_index)
        for b in utils.batch(row_index, _ubatch_size):
            n_step += 1
            # prep targets
            target_users = np.repeat(b, _nscores_user)
            target_users_rand = np.repeat(np.arange(len(b)), _nscores_user)
            target_items_rand = [np.random.choice(V_pref.shape[0], _nscores_user) for _ in b]
            target_items_rand = np.array(target_items_rand).flatten()
            target_ui_rand = np.transpose(np.vstack([target_users_rand, target_items_rand]))
            [target_scores, target_items, random_scores] = sess.run(
                [deepcf.tf_topk_vals, deepcf.tf_topk_inds, deepcf.preds_random],
                feed_dict={
                    deepcf.U_pref_tf: U_pref[b, :],
                    deepcf.V_pref_tf: V_pref,
                    deepcf.rand_target_ui: target_ui_rand
                }
            )
            # merge topN and randomN items per user
            target_scores = np.append(target_scores, random_scores)
            target_items = np.append(target_items, target_items_rand)
            target_users = np.append(target_users, target_users)

            tf.local_variables_initializer().run()
            n_targets = len(target_scores)
            bperm = np.random.permutation(n_targets)
            n_targets = min(n_targets, _max_data_per_step)
            dbatch = [(n, min(n + _dbatch_size, n_targets)) for n in xrange(0, n_targets, _dbatch_size)]
            f_batch = 0
            for (start, stop) in dbatch:
                dbatch_perm = bperm[start:stop]
                dbatch_u = target_users[dbatch_perm]
                dbatch_v = target_items[dbatch_perm]
                if _deepcf_dropout != 0:
                    n_to_drop = int(np.floor(_deepcf_dropout * len(dbatch_perm)))
                    dbatch_perm_perm1 = np.random.permutation(len(dbatch_perm))[:n_to_drop]
                    dbatch_perm_perm2 = np.random.permutation(len(dbatch_perm))[:n_to_drop]
                    dbatch_v_pref = np.copy(dbatch_v)
                    dbatch_u_pref = np.copy(dbatch_u)
                    dbatch_v_pref[dbatch_perm_perm1] = V_pref_last
                    dbatch_u_pref[dbatch_perm_perm2] = U_pref_last
                else:
                    dbatch_v_pref = dbatch_v
                    dbatch_u_pref = dbatch_u

                preds_out, _, loss_out = sess.run(
                    [deepcf.preds, deepcf.updates, deepcf.loss],
                    feed_dict={
                        deepcf.Uin: U_pref_expanded[dbatch_u_pref, :],
                        deepcf.Vin: V_pref_expanded[dbatch_v_pref, :],
                        deepcf.Ucontent: U_content[dbatch_u, :].todense(),
                        deepcf.Vcontent: V_content[dbatch_v, :].todense(),
                        #
                        deepcf.target: target_scores[dbatch_perm],
                        deepcf.lr_placeholder: _lr,
                        deepcf.phase: 1
                    }
                )
                f_batch += loss_out
                if np.isnan(f_batch):
                    raise Exception('f is nan')

            n_batch_trained += len(dbatch)
            if n_step % _decay_lr_every == 0:
                _lr = _lr_decay * _lr
                print('decayed lr:' + str(_lr))
            if n_step % _eval_every == 0:
                recall_warm = utils.batch_eval_recall(
                    sess, deepcf.eval_preds_warm, eval_feed_dict=deepcf.get_eval_dict,
                    recall_k=_recall_at, eval_data=eval_warm)
                recall_cold_user = utils.batch_eval_recall(
                    sess, deepcf.eval_preds_cold,
                    eval_feed_dict=deepcf.get_eval_dict,
                    recall_k=_recall_at, eval_data=eval_cold_user)
                recall_cold_item = utils.batch_eval_recall(
                    sess, deepcf.eval_preds_cold,
                    eval_feed_dict=deepcf.get_eval_dict,
                    recall_k=_recall_at, eval_data=eval_cold_item)

                # checkpoint
                if tf_saver is not None and np.sum(recall_warm + recall_cold_user + recall_cold_item) > np.sum(
                                        best_warm + best_cold_user + best_cold_item):
                    best_cold_user = recall_cold_user
                    best_cold_item = recall_cold_item
                    best_warm = recall_warm
                    best_step = n_step
                    save_path = tf_saver.save(sess, _tf_ckpt_file)

                timer.toc('%d [%d]b [%d]tot f=%.2f best[%d]' % (
                    n_step, len(dbatch), n_batch_trained, f_batch, best_step
                )).tic()
                print('\t%s\n\t%s\n\t%s' % (
                    ' '.join(['%.4f' % i for i in recall_warm]),
                    ' '.join(['%.4f' % i for i in recall_cold_user]),
                    ' '.join(['%.4f' % i for i in recall_cold_item])
                ))
                summary_vals = []
                for i, k in enumerate(_recall_at):
                    if k % 100 == 0:
                        summary_vals.extend([
                            tf.Summary.Value(tag="recall@" + str(k) + " warm", simple_value=recall_warm[i]),
                            tf.Summary.Value(tag="recall@" + str(k) + " cold_user", simple_value=recall_cold_user[i]),
                            tf.Summary.Value(tag="recall@" + str(k) + " cold_item", simple_value=recall_cold_item[i])
                        ])
                recall_summary = tf.Summary(value=summary_vals)
                if train_writer is not None:
                    train_writer.add_summary(recall_summary, n_step)
