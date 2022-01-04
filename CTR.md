

## CTR

怀总两篇文章 [blog](https://zhuanlan.zhihu.com/p/54822778) [blog](https://zhuanlan.zhihu.com/p/372048174)



--- 

**特征交互**的model

### FM

Steffen Rendle. - InICDM 2010 - Factorization machines. 

[simple code](http://shomy.top/2018/12/31/factorization-machine/)

* one-hot编码特征向量过于稀疏，维度大，特征空间大，神经网络参数多

* linear regression各个特征分量独立考虑，没有考虑特征与特征之间的相互关系
* MF can be seen as FM when only using user and item embedding.

LR: $y = w_0 + \sum_{i=1}^{n}  w_ix_i$

consider feature interaction: $y = w_0 + \sum_{i=1}^{n}  w_ix_i + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}w_{ij}x_ix_j$  

​	but $w_{ij}=0$ if $x_i = 0$ or $x_j=0$.

FM: $y = w_0 + \sum_{i=1}^{n}  w_ix_i + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}<v_i, v_j>x_ix_j$, where $v_i \in R^k$ but $O(kn^2)$ time. 

FM with O(kn) time:
$$
\begin{align}
\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}<v_i, v_j>x_ix_j &= 1/2(\sum_{i=1}^{n}\sum_{j=1}^{n}<v_i,v_j>x_ix_j - \sum_{i=1}^{n}<v_i,v_i>x_ix_i) \\
&= 1/2(\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_f v_{if} v_{jf}x_ix_j - \sum_{i=1}^{n}\sum_fv_{if}v_{if} x_ix_i)\\
&= 1/2 \sum_{f=1}^{k}((\sum_{i=1}^{n}v_{if}x_i)^2 -\sum_{i=1}^{n}(v_{if}x_i)^2)
\end{align}
$$




Weinan zhang et al. - ECIR 2016 - Deep Learning over Multi-field Categorical Data

Factorisation-machine supported Neural Networks [FNN] & SNN

对于输入x，x由多个field的特征拼接成，每个field采用one-hot特征编码，实际场景维度超级大

FNN是先用FM pretrain，得到每个field的一阶和二阶特征隐形量，作为DNN的输入

可以看作一种更为有效的embedding方式

FM捕获低阶特征，输入道DNN中提取高阶特征，有可能损失了低阶特征



FFM DeepFM ...

Huifeng Guo et al. - IJCAI 2017 - DeepFM: A Factorization-Machine based Neural Network for CTR Prediction华为诺亚方舟

[blog](https://www.jianshu.com/p/6f1c2643d31b) 强烈建议有时间，自己实现FM，FFM，DeepFM的CTR模型。以及使用libfm, libffm, xgboost/lightGBM(GBDT)等库。

相同输入，FM的输出y1，DNN输出y2，combine y1和y2(论文是y1+y2)，经过sigmoid作为模型预测

FM和DNN共享embedding层，对低阶特征和高阶特征单独建模。



---



**DIN.** Ali team. KDD2018 Deep interest network for click-through rate prediction

[blog](https://www.jianshu.com/p/7af364dcea12) [blog](https://zhuanlan.zhihu.com/p/54085498)   [知乎](https://zhuanlan.zhihu.com/p/51623339) [知乎](https://zhuanlan.zhihu.com/p/159727559)

1）用户行为历史，sum/mean pooling ---> attention  自适应地挑跟candidate相关的用户行为来做用户兴趣的表征

2）用GAUC这个离线metric替代AUC。 用Dice方法替代经典的PReLU激活函数。 介绍一种Adaptive的正则化方法 。介绍阿里的X-Deep Learning深度学习平台

previous method: Emebdding&MLP,

* user interests are captured from user behavior data

* use a fix-length vector to represent user diverse interest

using features: Table 1 in paper.

Difference between DIN and previous method:

* base model 使用embedding+pooling的方式，将user的历史行为映射为vector

* DIN: 根据候选ad的embedding，和user behavior的embedding，通过attention，算加权和，作为user behavior的embedding。这里attention，将[behavior embedding i, ad embedding j, i j element-wise product, i-j]. 输入MLP求attention权重，output layer没有softmax。就是所有权重和不要求为      

* DIN也尝试了LSTM来处理user behavior，但效果没有很大提升           

Mini-batch aware regularization:

* without regularization, easy to overfit
* with regularizaiton like l1, l2. 对于输入很稀疏的大规模网络，一个batch内只有很少非0的稀疏特征相关参数参与运算，l1/l2会对网络所有参数计算。计算负载很大。大量embedding参数。因此，提出只对batch内非0特征有关的参数来算正则项。

data adaptive activation func (Dice). 对PRelu的修改。使用了输入数据的均值方差，考虑数据分布。

Related work:

* using complex MLP network

* Deep Crossing: Web-scale modeling without manually crafted combinatorial features

* Wide & deep learning for recommender systems. 2016

* Deep neural networks for youtube recommendations. 2016

* DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. 2017





**DIEN** Deep Interest Evolution Network AAAI19 [知乎](https://zhuanlan.zhihu.com/p/50758485) [DIN&DIEN](https://zhuanlan.zhihu.com/p/78365283)

DIN基础上，加入两层GRU

1）对用户点击序列，使用GRU建模，每一时间步定义辅助loss：hidden state和下一个clk item embedding做内积当正样本，随机采样一个负样本，构建交叉熵loss

2）attention：target item和hidden state算attention权重，

3）下一层GRU和attention权重结合

* 直接对hidden state加权，做为下一层GRU的输入
* attention权重替换GRU中的update gate
* GRU update gate 乘 attention权重



MIMN

SIM
