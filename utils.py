import time
import datetime
import numpy as np
import scipy
import tensorflow as tf
from sklearn import preprocessing as prep


class timer(object):
    def __init__(self, name='default'):
        """Initialize a new `Stopwatch`, but do not start timing."""
        self._start_time = None
        self._name = name

    def tic(self):
        self._start_time = time.time()
        return self

    def toc(self, message):
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        print('[{0:s}] {1:s} elapsed [{2:s}]'.format(self._name, message, self._format(elapsed)))
        return self

    def reset(self):
        self._start_time = None
        return self

    def _format(self, s):
        delta = datetime.timedelta(seconds=s)
        d = datetime.datetime(1, 1, 1) + delta
        s = ''
        if (d.day - 1) > 0:
            s = s + '{:d} days'.format(d.day - 1)
        if d.hour > 0:
            s = s + '{:d} hr'.format(d.hour)
        if d.minute > 0:
            s = s + '{:d} min'.format(d.minute)
        s = s + '{:d} s'.format(d.second)
        return s


def batch(iterable, _n=1, drop=True):
    it_len = len(iterable)
    for ndx in range(0, it_len, _n):
        if ndx + _n < it_len:
            yield iterable[ndx:ndx + _n]
        elif drop is False:
            yield iterable[ndx:it_len]


def tfidf(x):
    x_idf = np.log(x.shape[0] - 1) - np.log(1 + np.asarray(np.sum(x > 0, axis=0)).ravel())
    x_idf = np.asarray(x_idf)
    x_idf_diag = scipy.sparse.lil_matrix((len(x_idf), len(x_idf)))
    x_idf_diag.setdiag(x_idf)
    x_tf = x.tocsr()
    x_tf.data = np.log(x_tf.data + 1)
    x_tfidf = x_tf * x_idf_diag
    return x_tfidf


def prep_standardize(x):
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def prep_standardize_dense(x):
    scaler = prep.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled


def batch_eval_recall(_sess, tf_eval, eval_feed_dict, recall_k, eval_data):
    tf_eval_preds_batch = []
    for (batch, (eval_start, eval_stop)) in enumerate(eval_data.eval_batch):
        tf_eval_preds = _sess.run(tf_eval,
                                  feed_dict=eval_feed_dict(
                                      batch, eval_start, eval_stop, eval_data))
        tf_eval_preds_batch.append(tf_eval_preds)
    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    tf.local_variables_initializer().run()

    # filter non-zero targets
    y_nz = [len(x) > 0 for x in eval_data.R_test_inf.rows]
    y_nz = np.arange(len(eval_data.R_test_inf.rows))[y_nz]

    preds_all = tf_eval_preds[y_nz, :]

    recall = []
    for at_k in recall_k:
        preds_k = preds_all[:, :at_k]
        y = eval_data.R_test_inf[y_nz, :]

        x = scipy.sparse.lil_matrix(y.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = y.multiply(x)
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
    return recall
