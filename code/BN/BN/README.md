[summar of normalization](https://zhuanlan.zhihu.com/p/33173246)

# Whiten

* PCA whiten: zero mean, one var, 
* ZCA whiten: zero mean, same var
* decorrelated.
* often applied to NN inputs.


# BatchNormalization

Google-2015-Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Advantages**
* reduce internal covariant shift, accelerate network convergence. use higher learning rate.
* reduce the dependency of params init valus
* as regularizer, reduce the dependency of dropout
* prevent network to saturated mode when us saturated activation func. (sigmoid, tanh)

**Towards Reducing Internal Covariate Shift**
* fix distribution of layer input. network converges faster if inputs are whitened.  [Paper](https://papers.nips.cc/paper/2011/file/e836d813fd184325132fca8edcdfb40e-Paper.pdf)
* directly apply whiten to each layer of DNN not work. 
> layer input u, bias b ===> x = u+b 
>
> whiten (centered): x' = x-E(x) = u+b - E(u+b), and E() not depend on b. ===> x' = u - E(u)
> SGD update b, but layer output x' not changed, loss not changed. (ecenter+scale worse)
* whiten is expensive. need cal Cov matrix and inverse sqrt root.
* whiten is not everywhere differentiable

**Simplifications ---> BN**
* normalize each dim of features independently. [not consider decorrelated.]
* use mean and var of batch smaples [batch setting using whole set is impractical]
* introduce trainable params $\gamma, \beta$, restore the representation power.
* BN顺序 (通常是全连接后，激活函数前。但也有研究说用在激活函数后效果更佳。)

**BN for CNN**
* BN do normalizaton for each dim of batch input features.
* For CNN processing imags, shape=[bs, channel, height, width]. treat channel as feature, 
  cal mean&var for bs\*height\*witdth values, and do BN. (one $\gamma, \beta$ for each map feature.)

**BN not for RNN**
* sequence length not fixed ---> LayerNormalization
* same dim of features are supposed to be from the same distribution.
* also not suitable for small batch size.

**Inference**

* 训练时每个batch mean var 的无偏估计 
* moving average [这个用的多，tf.layers.batch_normalization()中有moving_mean, moving_variance参数]

**Ref**
[Paper Notes](https://zhuanlan.zhihu.com/p/340856414) ｜ [原理与tf实战](https://zhuanlan.zhihu.com/p/34879333)  ｜[code理解BN](https://blog.csdn.net/shaozhenghan/article/details/81103838) ｜ [Paper&Code](https://zhuanlan.zhihu.com/p/50444499)

How Does Batch Normalization Help Optimization? - NIPS 2018 (deny ICS)



# LayerNormalization

2016 - layer Normalization

**BN**
* not suitable for RNN (not fixed seq length). online learning or small batch size 
* cal mean/var for each dim using bs values.(input of one layer: X shape=[bs, dim]) 

**LN**
* cal mean/var for all hidden units of one layer, cal mean/var for each sample using dim values.
* suitable for RNN,small batch size even only one sample.| same operations for train&test

**BN vs LN | Transformers use LN not BN**

* ICML2020-PowerNorm: Rethinking Batch Normalization in Transformers
* [notes](https://www.zhihu.com/question/395811291/answer/1257223002)
[notes](https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%886%EF%BC%89Normalization%E6%96%B9%E5%BC%8F/#layer-normalization-ln)
[notes](https://zhuanlan.zhihu.com/p/38755603)

**Question**

LN $\gamma, \beta$ has the same dimension of hidden states, which means different training samples share $\gamma, \beta$, same as BN. In BN, $\gamma, \beta$ can be seen as mean/var, not for LN.



# Code
## File
* data.py: get mnist dataset.

  ```python
  mnist = tf.keras.datasets.mnist # minist data is saved in the folder ~/.keras
  mnist.load_data()               # train | valid | test dataset.
  ```

* model.py: define FNN with/without BN/LN model for mnist. 

* main.py: train/test/hyper-params. | plot_exp_results.

* run.sh: different exp settings.

## Exp Settings
* BN (--ubn) 
  * large weight (--ulw)    [BNL]
  * small weight               [BNS]

* LN (--uln)
  * large weight (--ulw)    [LNL]
  * small weight               [LNS]

* No BN&LN
  * large weight (--ulw)    [NBNL]
  * small weight               [NBNS]


* tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    bn/moving_mean|moving_variance: zero-init|one-init
    without: moving_mean|moving_variance is not updated. still zero|one. (comment out the code in model.py) [BNL_noctrl | BNS_noctrl]
    with: moving_mean|moving_variance is updated.


## Exp Results
|          | BNS    | BNL    | NBNS   | NBNL   | BNS_noctrl | BNL_noctrl | LNL | LNS |
| -------- | ------ | ------ | ------ | ------ | ---------- | ---------- | ------ | ------ |
| Test ACC | 0.9816 | 0.9500 | 0.9760 | 0.1140 | 0.8045     | 0.5468     |  |  | 

./rst/learning_curve.pdf show the training curve of the above settings.



## TODO
(1)
> init = tf.truncated_normal_initializer(mean=10, stddev=0.1)
> init = tf.truncated_normal_initializer(mean=0, stddev=0.1)

(2)
> init = tf.truncated_normal_initializer(mean=0, stddev=10)
> init = tf.truncated_normal_initializer(mean=0, stddev=0.1)

(1) has bad training curve. training loss does not decrease.

tf.contrib.layers.layer_norm() import Error on MAC-conda-py27.

TODO: BN-to-git vimrc-git LN-exp-record

## Code Ref
BN: https://zhuanlan.zhihu.com/p/34879333
LN: https://blog.csdn.net/qq_34418352/article/details/105684488


