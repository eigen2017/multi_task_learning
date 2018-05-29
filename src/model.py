import tensorflow as tf


class MultiTaskFcModel:
    def __init__(self):
        self._construct_graph()

    def _construct_graph(self):
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.y = tf.placeholder(tf.float32, shape=[None, 33], name='y')
        self.x = tf.placeholder(tf.float32, shape=[None, 10], name='x')

        self.x_norm = self._batch_norm(self.x, name='x_norm')

        self.z1 = tf.layers.dense(inputs=self.x_norm, units=20, name='z1')
        self.z1_norm = self._batch_norm(self.z1, name='z1_norm')
        self.a1 = tf.nn.relu(self.z1_norm, name='a1')

        self.z2 = tf.layers.dense(inputs=self.a1, units=40, name='z2')
        self.z2_norm = self._batch_norm(self.z2, name='z2_norm')
        self.a2 = tf.nn.relu(self.z2_norm, name='a2')

        self.z3 = tf.layers.dense(inputs=self.a2, units=80, name='z3')
        self.z3_norm = self._batch_norm(self.z3, name='z3_norm')
        self.a3 = tf.nn.relu(self.z3_norm, name='a3')

        self.z4 = tf.layers.dense(inputs=self.a3, units=100, name='z4')
        self.z4_norm = self._batch_norm(self.z4, name='z4_norm')
        self.a4 = tf.nn.relu(self.z4_norm, name='a4')

        self.z5 = tf.layers.dense(inputs=self.a4, units=60, name='z5')
        self.z5_norm = self._batch_norm(self.z5, name='z5_norm')
        self.a5 = tf.nn.relu(self.z5_norm, name='a5')

        self.z6 = tf.layers.dense(inputs=self.a5, units=33, name='z6')
        self.z6_norm = self._batch_norm(self.z6, name='z6_norm')
        self.a6 = tf.nn.sigmoid(self.z6_norm, name='a6')

        self.my_prediction = self._get_predictions_from_float(self.a6, name='my_prediction')

        self.precision_rate = self._get_statistic_rates(self.my_prediction, self.y, name='precision_rate')
        self.recall_rate = self._get_statistic_rates(self.my_prediction, self.y, name='recall_rate')
        see_precision_rate = tf.identity(self.precision_rate, name='see_precision_rate')
        see_recall_rate = tf.identity(self.recall_rate, name='see_recall_rate')

        self.my_cost_compute = self._cost_compute(self.a6, self.y, name='my_cost_compute')

        self.my_optimize_op = tf.train.AdamOptimizer(1e-4).minimize(self.my_cost_compute, name='my_optimize_op')

    def _batch_norm(self, x, name):
        with tf.variable_scope(name):
            n_out = x.get_shape().as_list()[1:]
            beta = tf.Variable(tf.constant(0.0, shape=n_out), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=n_out), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(self.phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed_core = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
            normed = tf.identity(normed_core, name='normed')
        return normed

    def _cost_compute(self, yhat, y, name):
        with tf.variable_scope(name):
            lost_core = tf.multiply(-1.0, y * tf.log(yhat + 1e-7) + (1 - y) * tf.log(1 - yhat + 1e-7), name='lost_core')
            lost_batch = tf.reduce_mean(lost_core, axis=0, name='lost_batch')
            lost_stack = tf.reduce_sum(lost_batch, name='lost_stack')
            final_cost = tf.identity(lost_stack, name='final_cost')
        return final_cost

    def _get_statistic_rates(self, my_prediction, y, name):
        with tf.variable_scope(name):
            true_positive_cnts = tf.reduce_sum(tf.multiply(my_prediction, y), axis=0, name='true_positive_cnts')
            predicted_condition_positive_cnts = tf.reduce_sum(my_prediction, axis=0)
            condition_positive_cnts = tf.reduce_sum(y, axis=0)
            precision_rate = tf.div(true_positive_cnts, predicted_condition_positive_cnts + 1e-7)
            recall_rate = tf.div(true_positive_cnts, condition_positive_cnts + 1e-7)
            statistic_rates = tf.cond(tf.equal(name, 'precision_rate'), lambda:(tf.identity(precision_rate)), lambda:(tf.identity(recall_rate)), name='statistic_rates')
        return statistic_rates

    def _get_predictions_from_float(self, a6, name):
        with tf.variable_scope(name):
            my_greater = tf.greater(self.a6, 0.5, name='my_greater')
            my_float = tf.to_float(my_greater, name='my_float')
        return my_float


