import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from data import MnistData
from model import BNTestModel



def train(BNmodel, mnist, epochs, bs, save_path):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        bn_mean_list = sess.run(BNmodel.bn_mean_list)
        bn_var_list = sess.run(BNmodel.bn_var_list)
        print(bn_mean_list)
        print(bn_var_list)
        
        train_step = 0
        bset_val_acc = 0.
        for i in range(epochs):
            # train
            train_batches = mnist.train_batch(bs)
            for bx, by in train_batches:
                feed_dict = {BNmodel.tf_x:bx, BNmodel.tf_y:by, BNmodel.is_training:True}
                bloss, _ = sess.run([BNmodel.loss, BNmodel.train_op], feed_dict=feed_dict)
                train_step += 1
                
                if train_step % 10 == 0:
                    feed_dict = {BNmodel.tf_x:mnist.valid_x, BNmodel.tf_y:mnist.valid_y, BNmodel.is_training:False}
                    val_acc = sess.run(BNmodel.acc, feed_dict=feed_dict)
                    record = "Train Step:%d, Train Loss:%.4f, Val Acc:%.4f" % (train_step, bloss, val_acc)                   
                    print(record)
                    with open(os.path.join(save_path, 'rst.log'), 'a') as f:
                        f.write(record+'\n')
                    if bset_val_acc < val_acc:
                        saver.save(sess, os.path.join(save_path, 'model'))
        
        bn_mean_list = sess.run(BNmodel.bn_mean_list)
        bn_var_list = sess.run(BNmodel.bn_var_list)
        print(bn_mean_list)
        print(bn_var_list)


def test(BNmodel, mnist, restore):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, restore)
        feed_dict = {BNmodel.tf_x:mnist.test_x, BNmodel.tf_y:mnist.test_y, BNmodel.is_training:False}
        test_acc = sess.run(BNmodel.acc, feed_dict=feed_dict)
        print("test acc:%.4f" % test_acc)


def analysis(BNmodel, mnist, restore):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, restore)
        feed_dict = {BNmodel.tf_x:mnist.test_x, BNmodel.tf_y:mnist.test_y, BNmodel.is_training:False}
        test_acc = sess.run(BNmodel.acc, feed_dict=feed_dict)
        print("test acc:%.4f" % test_acc)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--h_dim', type=int, nargs='+', default=[128,128,10], help='FNN hidden dim')
    parse.add_argument('--ulw', action='store_true', help='whether use large weight for FNN')
    parse.add_argument('--ubn', action='store_true', help='wheter use batch normalization')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parse.add_argument('--bs', type=int, default=64, help='batch size')
    parse.add_argument('--epochs', type=int, default=60, help='training epochs')
    parse.add_argument('--save_path', type=str, default='./rst/test', help='saved model path')
    args = parse.parse_args()
   
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    mnist = MnistData()
    args.in_dim = mnist.input_dim
    args.out_dim = mnist.label_num
    BNmodel = BNTestModel(args) 
    
    #train(BNmodel, mnist, args.epochs, args.bs, args.save_path)
    test(BNmodel, mnist, os.path.join(args.save_path, 'model'))


def plot():
    folders = ['./rst/BNS', './rst/BNL', './rst/NBNS', './rst/NBNL', './rst/BNS_noctrl', './rst/BNL_noctrl']
    for folder in folders:
        with open(os.path.join(folder, 'rst.log'), 'r') as f:
            line = f.readline()
            steps, loss, acc = [], [], []
            while line:
                line = line.strip().split(',')
                line = [e.split(':')[1] for e in line]
                steps.append(int(line[0]))
                loss.append(float(line[1]))
                acc.append(float(line[2]))
                line = f.readline()
            plt.plot(steps, acc, label=folder)
    
    plt.legend(loc='lower right')
    plt.savefig('./rst/learning_curve.pdf')



if __name__ == '__main__':
    #main()
    plot()
