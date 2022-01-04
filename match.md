# Paper & knowledge

https://zhuanlan.zhihu.com/p/398041971 怀总屠龙少年

* 

* Deep Learning for Matching in Search and Recommendation Junxu, Xiangna He, Hang Li. [note](https://shimo.im/docs/PJXVHHGwKrWdk3rC)

* [zhihu](https://zhuanlan.zhihu.com/p/58160982)

* TODO: TDM ???

* AKNN

  [blog](http://xmanong.com/project/59396) [blog](https://www.it610.com/article/1281792594090672128.htm) [benchmark](http://ann-benchmarks.com/)

  * NN. nearest neighbor

  * ANN. approximate nearest neighbor

  * AKNN. approximate K nearest neighbor

  给定一个向量X=[x1,x2,x3...xn]，需要从海量的向量库中找到最相似的前K个向量

  * Faiss: [usage](https://zhuanlan.zhihu.com/p/67200902) [usage](https://zhuanlan.zhihu.com/p/133210698)

  * pyflann



## Alimama match

git magizoo 文档



展示广告/效果/品牌

* 商品 item 淘系商品

* 非商品 非淘系商品 店铺(落地页形式) 自定义url 短视频 直播 帖子 互动



## TODO

multi instance learning https://zhuanlan.zhihu.com/p/40812750

KL/JS的问题：https://zxth93.github.io/2017/09/27/KL%E6%95%A3%E5%BA%A6JS%E6%95%A3%E5%BA%A6Wasserstein%E8%B7%9D%E7%A6%BB/index.html

由此W距离的优势

[深度学习在阿里B2B电商推荐系统中的实践](https://mp.weixin.qq.com/s/OU_alEVLPyAKRjVlDv1o-w)

[微信「看一看」多模型内容策略与召回](https://mp.weixin.qq.com/s/s03njUVj1gHTOS0GSVgDLg)

[微信「看一看」 推荐排序技术](https://mp.weixin.qq.com/s/_hGIdl9Y7hWuRDeCmZUMmg)

GMV

PDN Path-based Deep Network for candidate item matching in recommenders - sigir21 阿里手淘 [blog](https://mp.weixin.qq.com/s/NznGxYghZZI-h_gnE2cu3A)

DFN





## 机器学习指标

confusion matrix: Y-axis是真实正负类，X-axis是预测的正负类 【所以confusion matrix的建立是依赖于threshold】



FPR = false position rate = $\frac{FP}{FP+TN}$, 预测为正的负样本，占真实负样本的比例。增大threshold，更少的样本被预测为正，FPR减小。

TNR：预测为负的负样本，占真实负样本的比例。threshold增大，TNR增大。

FNR = false negative rate = $\frac{FN}{TP+FN}$, 预测为负的正样本，占真实正样本的比例。增大threshold，更多样本被预测为负，FNR增大。

TPR【recall】：所有正样本中，被预测为正的比例。增大threshold，recall降低。

precision (PPV )：所有预测为正的样本中，真实正样本比例。【增大threshold，fp会比tp下降的更快，precision增大】

negative predict value NPV：所有预测为负的样本中，真实负样本的比例。【负样本的precision，增大threshold，NPV会下降】

Fscore [blog](https://www.jianshu.com/p/f7ea71f2344f)



acc: 真正+真负 在所有正负样本中的比例。增大threshold，acc变化趋势不一定。受数据不平衡的影响。



roc curve: x-axis是FPR，y-axis是TPR。auc为什么不受正负样本比的影响：在于横轴FPR只关注负样本，与正样本无关；纵轴TPR只关注正样本，与负样本无关。所以横纵轴都不受正负样本比例影响，积分当然也不受其影响。

https://zhuanlan.zhihu.com/p/79698237

http://sofasofa.io/forum_main_post.php?postid=1003723

https://blog.csdn.net/Leon_winter/article/details/104673047

https://zhuanlan.zhihu.com/p/34655990



PR curve：x-asix是recall，y-axis是precision





## Match

推荐技术演化 https://zhuanlan.zhihu.com/p/100019681

召回模型演化 https://zhuanlan.zhihu.com/p/97821040



### 启发式召回

协同过滤 

* 阿里ETREC

* Swing



#### BPRloss

Rendle et al. 2009 BPR: Bayesian personalized ranking from implicit feedback

​	[blog](https://medium.com/ai-academy-taiwan/bpr-與推薦系統排名-4beadf6e672d) [blog](http://lipixun.me/2018/01/22/bpr)

​	pairwise loss func. --- Analogies to AUC

​	AUC: 简单来说其实就是随机抽出一个正样本，一个负样本，然后用训练得到的分类器来对这两个样本进行预测，预测得到正样本的概率大于负样本概率的概率

​	AUC中delta不可求导，使用log sigmoid代替。max posterior优化参数提出了框架，对于user-item的评分可以使用之前的模型，例如MF

https://zhuanlan.zhihu.com/p/34757400

https://blog.csdn.net/ch_609583349/article/details/88308700



#### CF

**swing** [blog](https://thinkgamer.blog.csdn.net/article/details/115678598?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)

* 秋千：item i 被user u 和 v购买，三者组成一个秋千。 若u v同时也购买了item j，则i 和 j有一定相似性
* $sim(i, j) = \sum_{u \in U_i \and U_j} \sum_{v \in U_i\and U_j} \frac{1}{\alpha+|I_u\and I_v|} $. 含义就是，item i j相似性计算，是对所有同时购买了i j的user pair u和v，统计u v共同购买的item数目。共同购买item数目越小，u v越不相似，则i j越相似。反之，item i j相似性减小



**Matrix Factorization.** Koren et al. 2009 Matrix factorization techniques for recommender systems

[	blog](https://zhuanlan.zhihu.com/p/28577447)      

​	$\hat{r}_{ui} = u + b_i + b_u + q_i^Tp_u$
​	$min_{p*, q*, b*} \sum_{(u,i)\in K} (r_{ui}-\hat{r}_{ui})^2 + \lambda(||q_i||^2+||p_u||^2+b_u^2+b_i^2)$

**NCF.** He et al. www2017 Neural collaborative filtering

[	blog](https://blog.csdn.net/roger_royer/article/details/107943578?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.add_param_isCf)

Rendle et al. 2020 Neural Collaborative Filtering vs. Matrix Factorization Revisited 

[	blog](https://zhuanlan.zhihu.com/p/143161957) [MF vs NCF (dot product vs MLP)](https://www.zhihu.com/question/396722911)

​	dataset is large or embedding dim is small, otherwise dot product is likely a better choice.

Google. WSDM18. Latent Cross: Making Use of Context in Recurrent Recommender Systems

[	blog](https://zhuanlan.zhihu.com/p/55031992) [blog](https://blog.csdn.net/qq_40006058/article/details/100812846)

​	Section. 4.1 Modeling Low-Rank Relations. -- MLP在拟合特征之间交互关系低效 (与Rendle2020相似结论)

​	有m个特征（user， item， time），每个特征有N个取值。 为每个生成长度为r的embedding 向量。拼接输入一个单层MLP。预测他们的点积。发现增大网络	宽，可以更好拟合关系。但效率很差。因此考虑使用RNN

Xiangnan He et al. SIGIR2019 Relational Collaborative Filtering:Modeling Multiple Item Relations for Recommendation

---

**NGCF.** Xiang Wang, Xiangnan He et al. Sigir 2019. Neural Graph Collaborative filtering

[blog](https://zhuanlan.zhihu.com/p/150299081)

MF, NCF user/item embedding directly are used for interaction.NGCF refine the embeddings by propagating them on the user-item interaction graph

* First-order propagation:      

* high-order propagation:stack first-order propagation layer.

* prediction. suppose propagating with L layers, concat L user embeddings, concat L item embsuser-emb, item-emb, use inner product to predict.

* optimization. BPR loss

* Dropout. drop message & drop node      

* matrix form for user-item bipartite

---

**lightGCN.** Xiangnan He et al. SIGIR2020 LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

[blog](https://blog.csdn.net/qq_39388410/article/details/106970194) [blog](https://blog.csdn.net/hestendelin/article/details/107479266?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.pc_relevant_is_cache&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.pc_relevant_is_cache)

在NGCF上，简化。去掉激活层和特征变换矩阵。效果明显提升。         

NGCF是对不同层embedding进行拼接，本文引入加权求和。



implementation: LightGCN

user-item interaction matrix: $R \in \mathbf{R}^{m*n}$. m, n are user number and item number. $R[i][j]$=1 or 0.

build adj matrix: 
$$
A = 
\begin{bmatrix}%没有任何分隔符
       \mathbf{0} & R \\
       R^T & \mathbf{0} 
\end{bmatrix} \in \mathbf{R}^{(m+n) * (m+n)}
$$
norm adj matrix: $A = D^{-1/2}AD^{-1/2}$. $D$ is degree matrix of $A$. 

$A$ can stored using sparse matrix.

For users embedding $U\in R^{m*d}$ and items embedding $I \in R^{n*d}$.  concat them $E = concat(U, I, dim=0) \in R^{(m+n)*d}$

lightGCN layer: $E = torch.sparse.mm(A, E)$



simply aggregate 1-hop neighbors info. (without higher hop and itself.)



> lightgcn pytorch source code: https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data
>
> 
>
> model - PureMF - bpr_loss
>
> **# //TODO: BPRLoss use softplus. (some paper use log-sigmoid)**
>
> 
>
> **# //TODO: utils.UniformSample_original???**
>
> Suppose there are N users, M items. and K interactions <user, pos-item> for train.
>
> then at each train epoch, sample K users from N users (K >> N, so each user can be choose serveral times).
>
> for each user u in the smapled K users:
>
> get the user u's all positive items Pitems.
>
> and random choose one pos-item from Pitems.
>
> then random choose one neg-item from all items except items in Pitems.
>
> then get the train instance. <u, pos-item, neg-item>
>
> Finally, get K train instance. 
>
> 
>
> At each epoch, not every <user, pos-item> in trainset will be sampled (about half).
>
> which means, some <user, pos-item> can be sampled many times, then sample a neg-item for them.
>
> But still, it is possible sample the same neg-item for the same <user, pos-item> pair.
>
> Which means, train set for each epoch will have repeat instance.(but very less.)
>
> (May in this way, about half <user, pos-item> pair in trainset do not appear in a train epoch, so need more epochs to train model.
>
> in this paper, model need about 1000 epoch to converge.)
>
> (In this way, in each epoch, we can use the same <user, pos-item> for many times(with different neg-items))
>
> 
>
> Paper: NCF, different sample method.
>
> for each pair <user, pos-item> in K interactions of trainset:
>
> sample n(n=4 in the origin code) neg-items for the pair <user, pos-item>
>
> also n neg-items can be repeated, allought the probability is very low.
>
> this paper use binary cross entropy as loss, not BPRLoss







### 单向量双塔

几种双塔模型 [知乎](https://zhuanlan.zhihu.com/p/339116577)

DSSM双塔 [知乎](https://zhuanlan.zhihu.com/p/136253355)

**Youtube Recsys16** deep neural networks for youtube recommendations

[blog](https://blog.csdn.net/u014128608/article/details/109520766) [blog](https://zhuanlan.zhihu.com/p/52504407)

双塔：

* user侧三层Relu DNN，输出当作user vector
* item侧没有DNN，而是user vector算softmax时会有权重，当做item vector。就是另外做了一个item embedding matrix，和input layer侧的item embedding不一样。
* 双塔召回后来广泛应用。item vector生成思路没有广泛使用，一是可以直接用input layer侧的item embedding当作item vector，或者就是item侧也做一个DNN。

特征：有很多特征工程的做法可以借鉴。

metrics：离线precision, recall ｜ 在线CTR，watch-time等



**Youtube Recsys19** Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations 

​		metrics: 离线recall + 在线指标



**DeepMatch优化实践** Ali ATA

* 在双塔向量召回，对比了sample softmax 和 NCE, 发现sample softmax效果要好。[论文section3.1对比了两者](https://arxiv.org/pdf/1602.02410.pdf?spm=ata.21736010.0.0.1e4253f7PQJztZ&file=1602.02410.pdf)
* optimizer选择尝试



### 多向量

双塔模型天然缺陷：user/item vector 只有最后预测时才有交互，poly-encoder主要探索这个问题，构建多个user vector，和item vector在最后一层做交互。

MIND论文更早，在此问题基础上探索了多兴趣建模（是否真能捕捉多兴趣还需探索，但模型上也是和poly-encoder相似，通过多个user vector，增加和item vec的交互）

user/item vec交互缺失，从DNN角度看，不仅表现在次数上，还体现在深度上，DNN每一层不同表征都缺少交互。poly-encoder/MIND思路都只是在最后一层增多了交互次数，缺少DNN底层的交互建模。

* 尝试对user/item构建hierarchical的vector，在最后预测时分别交互，并融合。【很简单的方法，双塔DNN每一层hidden vec都拿出来，做内积预测，不知道计算资源上会增加多少】



**MIND**: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall - CIKM19 阿里手淘 

[blog](https://jishuin.proginn.com/p/763bfbd55e67) ATA

* 召回侧，通过capsule network+动态路由方法，获得多个user embedding。和poly-encoder思路有点像。

* train时，label-aware attention方式融合多个vec

* inference时，多个user embedding各自找相近的item embedding，可以从中取topK



**Lazada首页召回** Ali ATA

* 尝试了两种多向量召回：MIND & self attention生成多个user vector
* DFN (Deep Feedback Network): 使用了曝光未点击样本。在原来sample softmax loss上，加了triplet loss



**Poly-encoders:** architectures and pre-training strategies for fast and accurate multi-sentence scoring ICLR20 Facebook

这篇是做文本检索（对话匹配），涉及很多transformer，但检索思路和目前推荐召回侧相同。 [知乎](https://zhuanlan.zhihu.com/p/119444637)

* bi-encoder：双塔模型

* cross-encoder：fc模型（论文中是transformer）

* poly-encoder：双塔模型的基础上，把context侧用类似multi head attention思路，生成多个vec，在和candi侧塔生成的vec做attention

论文实验结果：poly-encoder提升了bi-encoder，不如cross-encoder的表现。推理速度远远高于cross-encoder，比bi-encoder慢一点

在双塔模型的基础上，保证训练推理速度。同时缓解双塔模型，只有在最后预测时才有user-item interaction。就是user侧生成多个向量，分别与item vec交互。



### graph embedding

**EGES** 阿里KDD2018 - Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba

这篇文章比较早。引入graph embedding，是对CF升级。对用户行为过的item，召回最相似的item。

* build graph（加权有向边）：用户历史行为序列，划分为session，session内，相邻两个item i，j，则认为item i 到 item j有一条有向边，对所有user的所有session统计，边出现频次当作边的权重

* graph embedding
  * BGE (Base Graph Embedding): 基于用户历史行为构建item graph，然后做graph embedding （DeepWalk）
    * DeepWalk: random walk (按照边权重，采样下一个node) + skip gram
  * GES (graph embedding with side info): 一些行为次数少或无行为的item，难以学好embedding，引入side info. skip-gram模型，输入侧加入side info。每一种side info都映射为embedding，然后mean pooling 聚合
  * EGES (enhanced graph embedding with side info): 不同side info加权，引入attention.把GES中mean pooling换成attention，weight是随机初始化并训练得到，使用时过一下softmax

* cold start items section3.3.2: 一个完全新的item，item graph中没有该item node，所以没有embedding可以从graph中学到。就用avg   side info embedding来代替。（同样的，对于冷启动item，也没有对应的weight，只能avg）
* 用户序列 ---> graph ---> 频率加权随机游走序列， 为什么不直接对用户序列做skip-gram？【随机游走，采样操作可能生成更加多样化的序列，（exploration）。有一定概率采样到冷门item，比直接用户序列，能缓解item长尾问题。但文章中，应该加一组实验，对比直接对用户序列做skip-gram】 [zhihu](https://zhuanlan.zhihu.com/p/70198918)



 Graph Convolutional Neural Networks for Web-Scale Recommender Systems



### 长短期兴趣建模

**SDM** CIKM19-Ali. SDM: Sequential Deep Matching Model for Online Large-scale Recommender System

在双塔结构下，user侧做的很复杂（lstm+attention）[知乎](https://zhuanlan.zhihu.com/p/388514802)

* session 短期兴趣。lstm + attention【multi-head self attention，多个兴趣点】
  * session内用户兴趣比较稳定，一般都是单一兴趣。但用户兴趣点（point of interest）是多样的。比如用户在某个session关注鞋子，那么他会在意鞋子的颜色，品牌，店铺，风格等等方面。 ===》 multi head self attention
  * section3.1 讲了session的生成规则
* long-term 用户在不同session间兴趣变化比较大。
* gate fusion module: merge long-term and short-term preference features.
  * input: user profile embedding & long-term representation & short term representation





### 双塔以外的模型

**TDM**

https://zhuanlan.zhihu.com/p/78941783

https://www.6aiq.com/article/1554659383706

https://github.com/alibaba/x-deeplearning/wiki/%E6%B7%B1%E5%BA%A6%E6%A0%91%E5%8C%B9%E9%85%8D%E6%A8%A1%E5%9E%8B(TDM)



**Deep retrieval** --- ByteDance

类似于soft聚类的思想，但还有很多疑惑。[知乎](https://zhuanlan.zhihu.com/p/260453374) [ICLR21reviewer](https://openreview.net/forum?id=85d8bg9RvDT) 





### 实践blog

58 同城：向量化召回上的深度学习实践 [blog](https://www.6aiq.com/article/1618011600160)

[推荐系统实践——Facebook & YouTube](https://qianshuang.github.io/2020/08/04/practice/) (CTR修正)

[github DeepMatch Model](https://github.com/shenweichen/DeepMatch) 







### 交叉特征

* 特征上。user feature, item feature 以及 交叉统计特征。

  * 阿里妈妈这边使用的hit类特征：可以认为是hard模式的attention结构，用ad的shop属性去hit用户历史行为过的shop list，如果命中了那么就说明历史用户有过直接行为，用行为id和频次来表示这个组合特征；如果没有命中特征就是空的，

  * 笛卡尔积构造的交叉特征

* 模型上，DNN特征交叉。PNN，FM，deepFM等，阿里CAN模型 [zhihu](https://zhuanlan.zhihu.com/p/287898562)



双塔无法使用交叉特征？

* 打分候选集item非常大，user和每个item统计交叉特征无法满足
* user embedding 和 item embedding可以分开计算，无法像DNN那样多层交叉。







### 采样

skip-above思想：用户点过的Item之上，没有点过的Item作为负例 （假设用户是从上往下浏览Item）

对收藏、购买的数据做上采样，即复制样本到数据里面



**Sample Optimization For Display Advertising** --- Baidu CIKM short 20

challenge:

1）sample selection (survival) bias or covariate shift: 用于训练的广告库，和推理时的广告库，有很大不同。

2）展现广告是有长尾效应。且大部分广告都没被点击，许多高频次广告被当作负样本。

3）展现未点击样本，不一定是真实负样本

4）点击样本少，稀疏。

entire space: all ads (expose and no-exp) 

positive set：showed&clicked

four different sampling methods:

1）weighted random negative sampling. 低频曝光ad随机采样，高频按照word2vec按频次采样。在采用word2vec方法采样，同时优化了空间占用。

2）Real-Negative Subsampling. 按照word2vec方式，对展现未点击的样本进行降采样(频次越高，被抛弃概率越大)，降采样后的展现未点击样本当作负样本。 （曝光频次高的ad，代表pctr或bid比较大，过分打压，有损利益）

3）Sample refinement with PU Learning. 展现未点击样本，包含reliable negative + potential positive. refine展现未点击样本，使其只包括reliable negative (positive unlabeled learning)

4）Fuzzy Positive Sample Augmentation. 缓解正样本稀疏问题。在未曝光的ad中，根据一定规则，作为负样本。

5）Sampling with Noise Contrastive Estimation (NCE)

​	2 3)都是在优化真实负样本集，背后思想和skip-above相同。1)是扩大了负样本空间，包括了未曝光的样本。多个blog中提到，展现未	曝光负采样，不如随机负采样。  [blog](https://www.jianshu.com/p/2f64f49e43cd)

metric:

离线： 

​	1）click_recall (先对每个user算，在所有user中取平均。和我们的很像，只是我们目前是session粒度的做)

​	2）cost_recall (hit到的ad的bid求和 / 所有user点击过的ad的bid求和，)

​	3）clk AUC

在线：CPM





**EBR** **Embedding-based Retrieval in Facebook Search** --- Facebook KDD20  [知乎](https://zhuanlan.zhihu.com/p/165064102) （召回侧负样本构造很重要）

Model: 

1) loss：和Youtube双塔做法不同，Facebook使用triplet loss来优化，模型输出不转为概率分布，而是<quey，item>的距离。

triplet loss （pairwise hinge loss）: 对所有三元组，正样本距离要小于负样本距离，当距离小到阈值m时，便不再优化。论文提到，m的值对结果影响挺大，在不同task上，也不同。
$$
L = \sum_i max(0, D(q, d^+)-D(q,d^-)+m)
$$
Youtube 使用sample softmax，也是在提高正样本得分，减小负样本得分

BPR loss：$L=-log Sigmoid(S^+-S^-), score:S$  BPR近似优化AUC。 背后都是Pairwise LTR



evalutation：

​	离线指标recall

​	Airbnb所使用的方法是看“用户实际点击”在“召回结果”中的平均位置

​	召回的离线评测难做，因为召回的结果未被用户点击，未必说明用户不喜欢，可能是由于该召回结果压根没有给用户曝光过

​	离线指标和在线指标有gap



2）特征工程 + 工程调优(fasis库的调优， serving)

3）Section2.4  正负样本构造研究，确定click当正样本，random sample当负样本

正样本：click

负样本：

random samples: random sample from doc pool. 不能等概率采样，按频次等方法。代表了easy cases，基本和query不匹配

non-click impression: in the same session, impression and non-click   代表hard cases，本身在一定程度上和query较为匹配

non-click impression 的recall远远低于random sample的方法。在召回场景下，面对的是所有doc库，只用展现未点击当负样本，并不代表inference时的数据空间。



正样本：

click：更能反应user的真实兴趣

impression：对ranking的近似，ranking排序高的样本，当作正样本。

在同样数据量时，两种方式recall相差不大

augmenting click-based training data with impression-based data：无效果增益，说明展现样本没有带来额外的信息，以及数据量增大也没有带来增益。

4）hard mining strategies

5）embedding ensemble



**MOBIUS**: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search --- 百度凤巢KDD19 

[blog](https://zhizhou-yu.github.io/2020/09/12/Mobius.html)

Motivation:

* matching和ranking侧，目标不一致。在matching侧引入后链路的业务指标（CPM等), 同时考虑优化relevance，CTR(CPM)

  这里CPM = CTR*Bid

* insufficiency of click history.  ranking侧PCTR预测时，样本是召回侧返回的，通常会是高频ad/query。在召回侧用同样方式训练时，高频query/ad，会预测较大的pctr,但实际上是bad case。（对于高频出现过的query或ad，容易预测出高的pctr。还有一点是应该也会有训练/测试样本bias问题，训练用展现点击/未点击，预测在全样本空间打分。）

Goal：

* bad case：relevance低，PCTR高
* 召回目标是，在保证relevance的前提下，召回最大化CPM的ad （类似多任务学习） [paper公式1和2对比]

method 【active learning】

* Teacher
  * 原先match层的model，Teacher （认为该model只关注relevance）

* Student
  * 原先rank层的CTRmodel （认为该model对于bad case处理不好）， 这里变为一个三分类网络，(click, unclick, bad case)，同时网络结构还是使用双塔。
  * 每个塔都会输出一个96dim vector，该vector切分为三个32dim的vector，三次内积，打三个分，分别代表三分类的logits，然后在softmax输出概率。

* data argument
  * 对于数据集中所有的query集合和ad集合，通过笛卡尔积来获得<query, ad>pair，然后用Teacher预测每个pair的relevance。
  * 取其中relevance低（threshold）的pair，用Student预测pctr, 取pctr高的样本(threshold)当作bad case，<query, ad, pctr>,存入train data buffer中。
  * buffer中会先加入user click history(展现点击/未点击)，后续用buffer中的数据来更新Student。

metric：离线AUC, average relevance.  在线CPM，CTR，APC(average click price)

conclusion

* 原来match侧，用点击样本训练，全局负采样，此时match双塔只关注relevance (用户点击与否，只跟relevance相关，与广告bid等无关).  所以当作Teacher
* 现在想直接在match侧训练CTRmodel，在使用rank PCTR模型的基础上，做了改进。训练/测试样本有gap。(motivation第二点) 所以通过query-ad的随机组合，扩充样本，同时只选择其中的bad case。随机组合的样本不一定是真实负样本，bad case是选择真实负样本。
* 方法上看，就是用展现点击/未点击样本训练match双塔，同时随机构造负样本，可以一定程度缓解训练/预测样本空间bias的问题。同时从中选择bad case，打压relevance低但CTR高的样本 （有点像两阶段构造负样本，娴晔也说过，美团尝试过两阶段加入hard 负样本，尽管收益很微弱）

* 展现样本：展现样本就是系统认为eCPM=CTR*bid高的样本。 ｜ 点击样本：user点击行为反映了relevance，没有广告主bid等指标。 可以对应上去 TODO//



match - rank ---》用一个CTRmodel，但原来relevance低的样本由match过滤，但一个CTRmodel不能处理这种情况。就设计了方法找到bad case。

match -- relevance

rank CTRmodel -- CTR

低relevance 低CTR --- match和rank判断一致

低relevance高CTR --- match可以去掉它。但只用一个model时，就是bad case



### metrics

[召回评估指标](http://yougth.top/2020/10/15/%E5%8F%AC%E5%9B%9E%E7%A6%BB%E7%BA%BF%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87/)

离线指标：

在线指标：除了业务相关的(ctr, ppc, cost等)，还有一个是展现占比





## 粗排

粗排模型COLD: [阿里粗排技术体系与最新进展分享](https://zhuanlan.zhihu.com/p/355828527) [COLD](https://zhuanlan.zhihu.com/p/186320100) [知乎](https://zhuanlan.zhihu.com/p/371139372)

将向量双塔模型，替换为精排模型 (7层DNN)，加入交叉特征。工程优化很重要。

特征选择（召回这边也有mmasi）：有对比过autofis里面的特征选择办法和这里的se block特征选择办法吗？我理解se block的特征重要性是每个样本不一样的，auto fis重要性与样本无关，不知道实际上的差别。对的，每个特征在每条样本上都会有一个特征重要性得分。这里最后会把每个特征在所有样本上的重要性求个平均，来代表这个特征的重要性，用于后面的特征选择。COLD第一次训练的时候有Se Layer，假如是M个feature group，也就是M组特征。基于特征重要性，最后选出了K个特征，K<M。后面的训练和线上，就没有Se Layer了，也只使用选出的K个特征。











## 知识蒸馏

简介 [blog](https://zhuanlan.zhihu.com/p/92166184)

label smoothing [blog](https://zhuanlan.zhihu.com/p/343807710) [blog](https://zhuanlan.zhihu.com/p/343988823)

蒸馏技术在推荐模型中的应用 [blog](https://zhuanlan.zhihu.com/p/386584493)

KDD 2020 淘宝召回 优势特征蒸馏 [blog](https://zhuanlan.zhihu.com/p/155935427)





## focal loss

https://zhuanlan.zhihu.com/p/49981234

https://zhuanlan.zhihu.com/p/32423092

多分类focal loss

https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

https://www.cnblogs.com/CheeseZH/p/13519206.html





## Muti Task

### 综述

[多目标学习在推荐系统中的应用](https://mp.weixin.qq.com/s?__biz=MzU2ODA0NTUyOQ==&mid=2247491211&idx=2&sn=a11007131f97835d655a4d451920843e&chksm=fc92a43dcbe52d2be58ef7e120d4bbff9ecdd8ffd5f9ebb473a97105c1fa71faadee1b403cf4&scene=126&sessionid=1605064054&key=059149d48d5c3e99fee7200bda4a5e4a7d0f1ab172f270b4a31ee39d0129a2210098dda57b4c275f69eb6ec5d674f4871ffcaef7636fa83bab1fb263f6c9673f88de8b4437ab0ab108b5e757060dc795c0031452e18002915f2f0c738c1f483eece0212fe66ba4aec07cd7b7fba4df7e812592e373fdc1c34e1bbf86d0acc1e1&ascene=1&uin=Mjg1NTU5MTQxMA==&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A77db8rvlMC6aDR5FrUFMBM=&pass_ticket=8hNub+Fu4yLIlzlFzkmkkQMUkX4moojyuksiXcSdcWti8q5+iG2QZTCpgM1wGGdz&wx_header=0)

[multi task 在推荐的实践](https://zhuanlan.zhihu.com/p/291406172)

[工业界推荐系统多目标预估的两种范式](https://zhuanlan.zhihu.com/p/125507748) MMOE 等参数共享模型 & ESMM 等任务依赖

https://www.zhihu.com/people/alex-zhai-19/posts?page=1



### sample reweight

蘑菇街首页推荐多目标优化之reweight实践：一把双刃剑 [blog](https://zhuanlan.zhihu.com/p/271858727)

阿里UC短视频 [blog](https://zhuanlan.zhihu.com/p/42777502)  [blog](https://mp.weixin.qq.com/s/FXlxT6qSridawZDIdGD1mw)

 [代码实践 ](https://zhuanlan.zhihu.com/p/337883819) [tf.nn.weighted_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits)

​	sample weight: 只有一个目标loss，只是对不同的样本赋予不同权重。模型简单，但其实并没有真正多任务学习，以及权重可能会放大一些样本带来噪声。





### 任务依赖

**ESMM**

[entire space multi-task model](https://blog.csdn.net/YyangWwei/article/details/116050894) 

Sample selection bias: 

​	CVR trained on impression&click dataset, used for impression data. (Entire space)

​	预估用户观察到曝光商品并且点击，之后购买此商品的概率

Data sparsity

[蘑菇街ESMM实践](https://zhuanlan.zhihu.com/p/76413089)

CTR / CTCVR 都可以在entire space训练

CTR的训练样本，大大多于CVR的训练样本，可以缓解数据稀疏问题。ESMM embedding层共享。

CVR原本输入是，展现且点击的样本特征，label是0，1（转化与否）

在ESMM中，两个子网络，CTR net和CVR net。 此时CVR net输入是展现样本的特征，假设每个展现样本都被点击了，输出预估的CVR，可见训练样本增多。但此时没有合适的CVR label，就用CTR*CVR得到CTCVR，CTCVR 和 CTR都可以从所有的展现样本中构建label。

Loss：所以有两个loss，一个是CTR的loss， 一个是CTCVR的loss。 CTCVR的loss可以约束学习CVR net。



**ESM2**： [blog](https://zhuanlan.zhihu.com/p/91285359)

Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction SIGIR20

在ESMM基础上，考虑favor/cart等行为

论文组合几个loss时，权重都是1



**NMTR** Neural Multi-Task Recommendation from Multi-Behavior Data

提了cascaded结构，B依赖于A任务，将A的输出作为B任务的输入。[训练trick](https://blog.taboola.com/deep-multi-task-learning-3-lessons-learned/)这个blog中有类似的方式介绍。



**Deep Bayesian Multi-Target Learning for Recommender System** [blog](https://blog.csdn.net/m0_52122378/article/details/111402369) [blog](https://zhuanlan.zhihu.com/p/74573041)

在不同task之间构建依赖关系，（B依赖A），A网络的输出vector就可以和Btask的feat拼接作为输入。思路和上一篇很像，也和ESMM, ESM2 有点像 (任务依赖关系，辅助target task学习)。



**美团猜你喜欢** [blog](https://tech.meituan.com/2018/03/29/recommend-dnn.html)

* missing value estimation
* KL boundary (和目前我们在召回侧做的多目标工作，有点像)



**AITM 美团 KDD21 Multi-task**: Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising

[blog](https://tech.meituan.com/2021/08/12/kdd-2021-aitm.html)

* 挑战/动机
  * 多步任务依赖：𝑖𝑚𝑝𝑟𝑒𝑠𝑠𝑖𝑜𝑛 → 𝑐𝑙𝑖𝑐𝑘 → 𝑎𝑝𝑝𝑙𝑖𝑐𝑎𝑡𝑖𝑜𝑛 → 𝑎𝑝𝑝𝑟𝑜𝑣𝑎𝑙 → 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛 	
  * positive feedback is sparser & delayed feedback of 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛：the positive feedback information of the former step to alleviate the class imbalance of the latter step. 

* related work
  * 底层网络共享：MMOE/PLE，没有显式建模任务依赖
  * ·输出层概率连乘：NMTR/ESMM/ESM2，概率标量乘法，忽视向量空间丰富的表征信息。一个任务预测不准，影响多个任务
* model Figure2(c)
  * logits前的vector，作为下一个task的输入。（经过MLP变换）
  * 前一个task输入向量和当前任务的向量，以self-attention方式得到最终向量。用于预测和下一个task的输入。
* loss
  * 每个任务cross entropy
  * 任务之间，前一个任务预测概率 > 后一个任务预测概率







### Parameters Sharing

* MOE：底层共享网络做成多个，称为experts，根据input+dnn+softmax得到权重(gate)，对每个expert的输出加权求和。多个模型的集成

* MMOE：MOE中，每个task的输入都是一样的。MMOE为每个task设置各自的gate. 不同task有不同的expert加权组合。

  [link](https://www.jianshu.com/p/0f3e40bfd3ce?utm_campaign=haruki&utm_content=note&utm_medium=seo_notes&utm_source=recommendation) [link](https://www.bilibili.com/read/cv6495744)	

  * Gate: 把输入input通过一个线性变换映射到nums_expert维，再算个softmax得到每个Expert的权重
  * Expert: 全连接网络，relu激活，每个Expert独立权重
  * [任务相关性实验](https://cloud.tencent.com/developer/article/1528474) MMOE论文生成数据，做了实验，任务之间相关性越强，模型学的越好。但没有做展示“跷跷板”现象的实验，PLE中有。

* PLE：将expert细分。每个task有独享的expert，所有task有一个共享的expert。

  Progressive Layered Extraction (PLE) --- Tencent PCG Recsys20

  论文中Figure1比较直观的总结了多任务学习中，共享结构的几种方法。

  [blog汇总](https://zhuanlan.zhihu.com/p/369272550) [blog](https://mp.weixin.qq.com/s/1ZZvEfQUDQat6nFnF67GcQ)

  * 每个任务的Loss加权和合并成一个Loss
  * 权重dynamic，给一个初值，随训练step变化



SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-task Learning https://zhuanlan.zhihu.com/p/150584473



**[与Attention的联系](https://zhuanlan.zhihu.com/p/125145283)** 

MMOE中的gate，就是对多个expert加权融合。和attention机制相似。

文章里做了一件有意思的事情：每个task设置一个context向量当作query，（随机初始化，随着模型训练。）key/value是每个expert的输出。query和key计算weight，在对value加权求和。这么做，在他们的场景下，结果优于MMOE。



**STAR:** One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction --- Alimama 2021

[zhihu](https://zhuanlan.zhihu.com/p/361113529)

多域(场景)建模 & 多任务学习：多域问题解决不同域的同一个问题，比如预测不同域的点击率。多任务优化目标是多个的，如同时优化点击率和转化率。现有多任务学习框架都是共享底层，不同的任务用不同的输出层，对于多域问题，并不能充分挖掘不同域之间的关系。

每个场景下训练一个model，浪费资源且数据量少 ---> 训练一个model，来servering所有的域，同时考虑不同域数据分布差异

* PN: 对BN修改。BN对同一分布下的数据统计均值方差，做归一化。显然多域问题下数据来自不同的分布。PN就是对不同域数据，统计各自的均值方差来做BN。

* star  topology FCN：所有domain有一个share FCN，每个domain有自己的specific FCN。因此若有M格domain，则有M+1个FCN，因为主要参数都在embedding部分，这块多个FCN带来的参数增加基本可以忽略。输入会经过share FCN和自己的specific FCN，两个网络的输出做融合。 这边有不同的融合方式吧，论文中给出的方法是，FCN每一层的weight 和 bias 融合，（weight逐元素乘，bias相加）。但也可以尝试其他方式的融合。【这块神似PLE】

* auxiliary network (论文用了2层FCN)

  inpu是domain indicator 以及其他能描述domain的特征。output是个标量，和star topology FCN的输出的logits相加，表示最终logits，通过sigmoid来表示pctr。 【这样加强了domain自己特征对最后输出的影响】

实验对比了MMOE，有效果，没有对比PLE。这篇文章好像和PLE挺像的，感觉是在PLE思路上，加入了domain自己特征的残差连接。





### Loss Weight

**naive method**

先给困难任务分配一个较大权重，简单任务分配一个较小权重，已使困难样本优化的更快.

multi-losses除以平均值获得各个loss的权重



**UWL**

多任务的loss相差很大，自动权重调节 [blog](https://zhuanlan.zhihu.com/p/269162365)

regression loss $L_1$, classification loss $L_2$. $s_1 = log\sigma_1^2$, $s_2=log\sigma_2^2$ . Weighted Loss $L$:
$$
L = 0.5*exp(-s_1) * L_1 + exp(-s_2)*L_2 + s1+s2
$$
$s_1, s_2$都为随机初始化，可学习参数。 所以这么做，和直接设可学习的权重+权重正则，有区别吗？（权重正则是必要的，不然权重都学习为0，最后loss直接为0）

方差大，不确定度大的，权重小。



**多任务学习在美图个性化推荐的近期实践** [blog](https://zhuanlan.zhihu.com/p/89401911)

样本reweight （类似focal loss）

UWL weighted Loss ： 估计出来的不确定性不稳定，参数容易学飘，甚至出现相反的结论，导致实验效果波动较大。在实践中，笔者采用效果正向提升的几天的不确定性值作为最终的参数。



**BIGO | 内容流多目标排序优化**[blog](https://mp.weixin.qq.com/s/3AMW-vUr2S9FBSDUr_JhpA)

loss weight 自动寻优，有用到RL的思想，但没有上RL模型



**optimization for MTL**

[survey paper](https://arxiv.org/pdf/2004.13379.pdf) 中有一部分是将optimization for MTL， [blog](https://zhuanlan.zhihu.com/p/269492239)参考改论文做了总结。



**Pareto-Efficient**

dominates

Pareto Efficient/optimal

Pareto Frontier

scalarization方法，对各个loss线性加权组合，要选择合适的系数，保证问题解是帕累托最优

Pareto stationary：满足KKT条件



**MGDA-NIPS18** Multi-Task Learning as Multi-Objective Optimization

[blog](https://zhuanlan.zhihu.com/p/68846373) [blog](https://blog.csdn.net/icylling/article/details/86249462)

基于scalarization method的Multiple gradient descent algorithm (MGDA)模型

1）对非共享参数做梯度下降

2）优化带约束的凸函数（基于KKT条件），求出weights，对共享参数梯度下降

3）2）步中每个任务都要求对共享参数的梯度，该论文做了转换，只求对共享参数的最有一层输出求梯度。



**Wechat WWW21** Personalized Approximate Pareto-Efficient Recommendation

[blog](https://cloud.tencent.com/developer/article/1816300)

Personalized Approximate Pareto-Efficient Recommendation - Wechat www2021

用户的目标级别的个性化需求，在user侧个性化

用RL (DDPG) 的方法，action为loss的权重，reward为各个loss梯度加权取反（Pareto KKT）, state考虑user以达到个性化。



**Ali-RecSys19** A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation

[zhihu](https://zhuanlan.zhihu.com/p/125598358)

根据Pareto efficient 满足的KKT条件，solve the complex Quadratic Programming problem

其中考虑了权重是有最小bound的

Pareto Optimal更有理论保证，但求解权重的过程太复杂。





### 训练

[训练trick](https://blog.taboola.com/deep-multi-task-learning-3-lessons-learned/)

交替训练：多个task的输入不同。taskA优化时，不会影响taskB的tower

一个Adam对总loss进行衰减，与两个Adam对两个loss分别进行衰减是不一样的

[trick](https://zhuanlan.zhihu.com/p/56613537): 加权求和loss / 不同学习率交替训练 / taskA的输出可以作为taskB的输入，反向时记得tf.stop_gradient()



**阿里 DUPN** Perceive Your Users in Depth: Learning Universal User Representations from Multiple E-commerce Task KDD18

​	重点在user representation（LSTM+attention）

​	multi task共享user representation

​	[blog](https://developer.aliyun.com/article/568166) ---> trick: BN离线训练，在线servering的一致性问题。



**MTL & 迁移学习**

MTL是机器学习的一种范式，属于迁移学习的范畴，但与大家通常理解的迁移学习的区别在于多任务学习是要借助其它的任务来帮助提升所有任务而非某一个任务的学习效果。

目前做的召回侧的MTL，理论上CTR 和 exp模型输出相乘，是我们需要的未归一化的概率。我们的思路就是用这两个模型的loss，一起帮助clk模型学习，其中target就是clk 模型，另外两个模型学的准确，应该是更有利于clk模型的学习。但其实我们只需要clk模型更好学习，不关系exp CTR模型学的如何。是否更类似迁移学习：1）预训练exp ctr模型，使其尽可能准确。2）想办法把exp ctr模型学到的知识迁移到clk模型上

但按照目前实验结果，单独train exp和四个loss联合得到的exp model，联合训练的exp效果更好。





## 广告

Expose (impression) - click - favcart - conversion

feature x， x 是高维稀疏的特征向量, click y, conversion z

pCTR p(y=1/x)	展现样本，点击1，未点击0

pCVR p(z=1/x, y=1)	展现样本且点击，转化1， 未转化0

pCTCVR p(y=1, z=1 / x)	展现样本，点击且转化1， 未点击或点击未转化 0

pCTCVR = pCTR * pCVR

CVR和CTCVR的区别：用户未点击，则一定不存在转化，这是CTCVR关注的事情。 用户未点击，可能是一些额外因素，比如商品展示图丑，但这个商品本身是足够符合用户兴趣，即假设用户点击了，则对应的CVR会比较大。

即对于未点击商品，其pCTCVR=0， 但其CVR不一定。也就是CTR和CVR之间，是比较独立的。CTR低，CTCVR肯定低，但CVR不一定低。



---

阿里妈妈的广告，主要是针对手淘，支付宝的大部分广告位有自己的团队去做。

展示广告

​	首猜（淘宝首页，猜你喜欢）：目的是增加点击。

​	首焦：淘宝首页，猜你喜欢，有三帧。（目的是增加曝光）

​	超推，购后，钻展

搜索广告（直通车，淘宝搜索框）：用户会输入一个query，相比之下，user侧的特征用的较少。（场景，两个user输入同样的关键词，淘宝搜索返回的结果基本一致）



pv（page view）: 跟具体业务逻辑有关，有的是用户浏览了就算一次，有的是用户点进详情页才算。

user profile: 用户性别年龄地域，购买力等静态特征（推荐组常用，广告这边较少用，之前match模型没有考虑，麓老板他们在尝试）

UT日志（userTrack): 以手淘为例，用户每次浏览，都会有对应的记录。

​	以日志形式存储：

* 曝光
* 点击 --> 详情页点击

​    下面信息是直接存储在数据库：

* 收藏：用户可能会收藏或取消收藏，（涉及状态变化的）
* 购买

UB：UT的一个子集，userTrack记录的内容可读性很差，

​	详情页点击（pv ，click) （UT日志）

​	fav，cart, buy （来自数据库）

feature：一部分ad曝光未点击，需要从UT中读取，同时考虑UB中的特征。



特征主要分为user侧，和广告主侧。

sample：（user， item）

user feat: 历史记录，历史记录的聚合，或过去一段时间的特征聚合。（match这边好像大部分都是简单的聚合函数，sum/mean pooling，没有上RNN）

user的一个请求，被系统自动标注为一个session，一个session内，有多个ad曝光。从中可以提取ad特征，

​	pv，

​	clk，    点击服务器

​	favcart，buy。 数据库

buy了广告才会扣钱，广告主关心投入产出比（ROI，投资回报比），rev/cost，越大越好。广告主的cost，就是alimama的收入。

结算服务器：广告主对每个广告都设的有投入资金，当钱花完了，广告需要下线，就会在索引数据库中标识。



user ---> tag ---> ad

Member 广告主

​	多个计划campaign --- 目的是提高 click，favcart，buy

​	adgroup --- 多个**定向**（ad ---> user）, 喜爱宝贝，相似宝贝，智能定向(一般模型学出来但没有明确解释)等

(注意广告并不指代某个实体，概念比较抽象)

tag: item，shop，category，或者组合等。    

ad--->tag 正排

tag--->ad 倒排

索引数据库：主要包括倒排，和正排表



所以召回做的事情，就是user--->tag的映射，各种召回算法。

​	简单的，例，直接user，找其感兴趣的shop当作tag，从索引数据库中返回对应的ad list

​	向量检索模型，通常返回数百级（500）的tag，索引数据库中可以返回万级别的ad

召回 - 排序 （粗排/精排）- 重排（策略） 



u-u graph, （定向）

user--->ad

p(a|u)  ice

​	(u, a) 正样本，采负样本，向量模型

p(i | u), mprm， asi adaptive scene interest





---

**复杂模型全库检索项目**

业界match现状：向量检索（双塔） + 在线ANN索引&检索方案（Faiss，Proxima ...）

复杂模型，更好的效果，充分利用算力

Relevance Proximity Graphs for Fast Relevance Retrieval AAAI2020

​	构建一张大图，图上检索

任意复杂模型+高效在线索引&检索 （预期提效RPM+5%）



向量模型：向量内积相似度

TDM：依赖索引（自由度低，推广范围没有向量模型广泛）。可以对足够复杂的模型，进行全库检索。

user seq的attention，消耗算力 ---> linear attention

local索引结构

索引构建：i2i similarity, 

​	forward_idx, inverted_idx两数组确定节点关系

模型样本：三层DNN+行为窗口target-attention，item侧只有item_id

​	每层NS采样，暂不支持nce

​	训练用首猜展现样本，测试是下一天同分布样本



量化：模型上线时float32--->int8    [int8量化](https://zhuanlan.zhihu.com/p/58182172)





**u2i**召回和**i2i**召回: **u2i**召回时，拿user的向量去召回topk个item，**i2i**召回时拿item的向量去召回topk个item。

hitrate的具体计算方法为，假设真实trigger（u2i召回时为user，i2i召回时为item）的**关联item集合为M**，而实际召回了top k个和trigger相似的items，若其中落在了**M**里的集合为**N**，可计算recall和precision。







---

本地文件 ---> odps

1）上传csv到服务器

2）d2上写sql，建表及分区

3）服务器上odpscmd，上传csv到建的表及分区中



---

## xdl

dataio.py



自己定义要读取的特征：

```python
"""
k:v --- real feature group
v   --- bin feature group

odps bwt 中，header列名会有 _0(nocommon) _1(common)的后缀
"""

# common real
common_real_fg = ["0_201_1", "0_203_1", "0_204_1"]
common_real_fg_k = ["k_" + i for i in common_real_fg]
common_real_fg_v = ["v_" + i for i in common_real_fg]

# common bin
common_bin_fg = ["1_001", "1_002"] #"1_005", "0_701"]

# ncommon
nocommon_bin_fg = ["2_001", "2_003", "2_004"] 
```



dataio.py

```python
for name in common_real_fg_k+common_real_fg_v+common_bin_fg+nocommon_bin_fg:
    io.feature(name, paiio.DataType.VarLen)

# ....
io_hook = io.get_hook(task_index, task_num, save_interval=None, filenames=paths)
raw_batch = io.read()

"""
raw_batch = {{nocommon_fg}, {common_fg}}

nocommon_fg/common_fg = 
{
 'k_0_204_1': ((arr1, arr2),)
 'v_0_204_1': ((arr1, arr2),)
 '1_001':    ((arr1, arr2),)
 
 '_indicator': ((arr1,),)
 'sample_id':
 'sample_tag':
 'mock_weight_tag':
 'weight_tag':
 'label_ext':
}
"""
```



```
nocommon:
((39899,), (40001,))
(40000, 7)
((39899,), (40001,))
((40000,), (40001,))
(40000, 1)
(40000, 1)
((40000,), (40001,))
(40000,)

common: 
((1772,), (2001,))
((1772,), (2001,))
((38954,), (2001,))
((38954,), (2001,))
((2242,), (2001,))
((2242,), (2001,))
((2000,), (2001,))
((33723,), (2001,))
(40000,)  ------------> indicator, [0,1,2...,1999]每个重复20次，正负样本共享user侧feature

```





test:

conf/asi/train/model/asi_esmm_lyf/xdl/model_test.py



user_table 

ad_table (fake) : 用来得到sample_id (as target_id) 和 ad_output (embedding+MLP ---> 64 dim feat)





user_table: (user_batch_size=250)

```
 nocommon: 
(u'2_004', (248,), (251,))
(u'label_ext', (250, 7))
(u'2_003', (248,), (251,))
(u'2_001', (250,), (251,))
('mock_weight_tag', (250, 1))
(u'weight_tag', (250, 1))
('sample_tag', (250,), (251,))
(u'sample_id', (250,))

 common: 
(u'k_0_204_1', (23,), (70,))
(u'v_0_201_1', (957,), (70,))
(u'k_0_203_1', (76,), (70,))
(u'v_0_204_1', (23,), (70,))
(u'v_0_203_1', (76,), (70,))
(u'1_001', (69,), (70,))
(u'1_002', (1349,), (70,))
('_indicator', (250,))
(u'k_0_201_1', (957,), (70,))
===================


 nocommon: 
(u'2_004', (249,), (251,))
(u'label_ext', (250, 7))
(u'2_003', (249,), (251,))
(u'2_001', (250,), (251,))
('mock_weight_tag', (250, 1))
(u'weight_tag', (250, 1))
('sample_tag', (250,), (251,))
(u'sample_id', (250,))

 common: 
(u'k_0_204_1', (50,), (64,))
(u'v_0_201_1', (937,), (64,))
(u'k_0_203_1', (43,), (64,))
(u'v_0_204_1', (50,), (64,))
(u'v_0_203_1', (43,), (64,))
(u'1_001', (63,), (64,))
(u'1_002', (1191,), (64,))
('_indicator', (250,))
(u'k_0_201_1', (937,), (64,))
===================

 nocommon: 
(u'2_004', (247,), (251,))
(u'label_ext', (250, 7))
(u'2_003', (247,), (251,))
(u'2_001', (250,), (251,))
('mock_weight_tag', (250, 1))
(u'weight_tag', (250, 1))
('sample_tag', (250,), (251,))
(u'sample_id', (250,))

 common: 
(u'k_0_204_1', (41,), (81,))
(u'v_0_201_1', (959,), (81,))
(u'k_0_203_1', (70,), (81,))
(u'v_0_204_1', (41,), (81,))
(u'v_0_203_1', (70,), (81,))
(u'1_001', (80,), (81,))
(u'1_002', (1569,), (81,))
('_indicator', (250,))
(u'k_0_201_1', (959,), (81,))
===================
```

---

ad_table: (batch_size=40000)

 nocommon: 

(u'2_004', (39631,), (40001,))

(u'weight_tag', (40000, 1))

(u'2_003', (39631,), (40001,))

(u'2_001', (40000,), (40001,))

('mock_weight_tag', (40000, 1))

('sample_tag', (40000,), (40001,))

(u'sample_id', (40000,))

(u'label', (40000, 2))



 common: 

(u'k_0_204_1', (0,), (40001,))

(u'v_0_201_1', (0,), (40001,))

(u'k_0_203_1', (0,), (40001,))

(u'v_0_204_1', (0,), (40001,))

(u'v_0_203_1', (0,), (40001,))

(u'1_001', (40000,), (40001,))

(u'1_002', (0,), (40001,))

('_indicator', (40000,))

(u'k_0_201_1', (0,), (40001,))



---





全部广告候选集：（考虑广告主各种诉求）

定向：广告主可以选择特定人群等。 ad <===> 一组user 

​	智能定向

​	自定义：重定向（retargeting），包含行为过特定或相似内容(实体)的人群

​					关键词，行为与某些关键词相关的人群

​					人口属性，基于年龄，性别等组合的人群



自定义：

user ---截断优选----> tag  ---截断优选----> ad

​	user只有一个tag，但tag很热门，可能召回很多ad

​	user很活跃，有很多tag，召回很多ad，且有可能长尾 

​	未截断后的样本数目，是截断的 5～10倍，所以这种方式难以逼近最优

先召回，后反向挂载

大人检：无截断，offline算，ecpm排序，线上查询队列。

智能定向： tag--->ad （拓展，shop item ad keyword）

​					user--->ad 规则拓展，模型计算(TDM, 向量模型)， 离线模型优选(广告给user打分)



bp 广告管理平台



item ---> keyword

相似关系：item1 ～ item2， 共现词







在线：

user---tag:

​					 k-v. 用户-taglist

​					k-k/v-v. 用户k--trigger 延伸出其他key （相似item/user，从属关系-店铺等，） - taglist  	

​					model: topK tag

重定向retargeting k-v, k-k,

相似：v-v. 如何计算相似

model：样本，特征，训练，预测，test，online inference





数据

数据库 维表：静态特征，淘宝用户表/类目表/广告主表等 --- 从数据中台拿表 tbcdm （cdm 中台）

行为表：

​	广告数据（pvclk， 参竞日志），

​	自然流量

​				UB user behavior: 工程团队整合了用户在淘系常见的行为

​					ecpm_algo_n_shark _ behavior_raw

​					dpv 详情页曝光（pv， 来自UT）, 【cart, buy, fav 来自数据库】 

​				UT（埋点数据）, dwd_user_track_v odps表 s_user_track

​						事件类型（页面，曝光，点击埋点）

（UT，数据库是所有数据的上源）

数据源：原始类数据（pv clk日志，UT+广告日志，没有人处理过的原始记录数据）		

​			   可信任数据：维表，UB，自己产的---通用型，扩展性 正确性，数据清洗验证，效率



广告 --- 广告中间层

UT --- 中间屏蔽层 自己的UT fund_n_sphinx_basic_user_track_log. scene分区字段

UB --- 中间屏蔽层 our UB.  

​	 unicorn: fund_rec_n_unicorn_basic_behavior   key_type字段 （user， key_ID， key_type）

​	delta: 滑动窗口

​			fund_rec_n_dragon_feature_user_basic  fea_name分区字段

​			git/dragon ---> feature_map.md    sql/delta/delta_user.sql

​			fund_rec_n_dragon_feature_user_delta.  fea_name=***_n 就是delta = f(T)-f(T-n)



k-v & k-k-v

相似： 协同过滤（阿里ETREC）， SWING 两个item share one pair users，向量相似

​			fund_rec_n_unicorn_swing/etrec/proxima_input/output

从属

​	非关键词：维表能拿到

​	关键词 item-->node

model: thestral 整体框架

​		样本：fund_rec_n_thestral_raw_sample_train  （仿照rank组） version分区字段

​						stag（sample tag 采样用）， pv id一般是session id， unit id 做什么检索就是什么id

​						kvlist上下文特征

​		特征：user侧 user_feat,  unit侧 unit feature table

