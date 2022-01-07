import numpy as np
import tensorflow as tf


class MnistData():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        train_x = train_x.reshape([train_x.shape[0], -1])
        test_x = test_x.reshape([test_x.shape[0], -1])
        train_x, test_x = train_x / 255., test_x / 255.

        train_x = train_x.astype(np.float32)
        test_x = test_x.astype(np.float32)
        train_y = train_y.astype(np.int32)
        test_y = test_y.astype(np.int32)
        
        test_num = test_x.shape[0]
        self.valid_x, self.valid_y = train_x[-test_num:], train_y[-test_num:]
        self.train_x, self.train_y = train_x[:-test_num], train_y[:-test_num]
        self.test_x, self.test_y = test_x, test_y
        self.input_dim = train_x.shape[1]
        self.label_num = train_y.max()+1
    
    def train_batch(self, bs):
        i = 0
        while i < len(self.train_x):
            batch_x = self.train_x[i:i+bs]
            batch_y = self.train_y[i:i+bs]
            i += bs
            yield batch_x, batch_y


if __name__ == '__main__':
    mnist = MnistData()
    train_batches = mnist.train_batch(bs=256)
    cnt = 0
    for bx, by in train_batches:
        print(bx.shape, by.shape)
        cnt += bx.shape[0]
    print(cnt, mnist.train_x.shape)
    

