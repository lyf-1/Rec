import tensorflow as tf


class BNTestModel():
    def __init__(self, args):
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim 
        self.h_dim = args.h_dim   # h_dim = [128, 128, 128]   
        self.ulw = args.ulw        #ulw: use_large_weight
        self.ubn = args.ubn        #ubn: use_batch_normalization
        self.lr = args.lr

        self._build_placeholder()
        self._build_graph()

    def _build_placeholder(self):
        self.tf_x = tf.placeholder(tf.float32, [None, self.in_dim], name='tf_x')
        self.tf_y = tf.placeholder(tf.int32, [None], name='tf_y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def _build_graph(self):
        if self.ulw:
            init = tf.truncated_normal_initializer(mean=0, stddev=10.)
        else:
            init = tf.truncated_normal_initializer(mean=0, stddev=0.1)

        x = self.tf_x
        for i, h_dim in enumerate(self.h_dim):
            x = tf.layers.dense(x, units=h_dim, kernel_initializer=init, name='fc'+str(i))
            if self.ubn:
                x = tf.layers.batch_normalization(x, name='bn'+str(i), training=self.is_training)
            x = tf.nn.relu(x)

        out = tf.layers.dense(x, units=self.out_dim, kernel_initializer=init, name='output')
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tf_y, logits=out))
        correct_pred = tf.equal(tf.argmax(out, axis=1, output_type=tf.int32), self.tf_y)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        '''
        if self.ubn:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        else:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        # tvars = tf.trainable_variables()
        saver = tf.train.Saver()
        avars = saver._var_list
        self.bn_mean_list = []
        self.bn_var_list = []
        for var in avars:
            if 'moving_mean' in var.name:
                self.bn_mean_list.append(var)
            if 'moving_variance' in var.name:
                self.bn_var_list.append(var)
