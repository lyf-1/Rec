# Survey

https://zhuanlan.zhihu.com/p/30720579

http://kubicode.me/2018/09/19/Deep%20Learning/GRU4REC-Session-Based-Recommendation/



Sequential Recommender Systems Challenges, Progress and Prospects IJCAI2019

SRS: given a interaction sequence $S=\{i_1, i_2..., i_{|S|}\}$.  $R=argmax f(S)$.

​	$i_j = <u, a, v>$

​	$u$: user | $v$: item --> may associated with some features.

​	$a$: action. --> types: click, purchase, add to cart ...  context: time, location ,weather...

​	$R$ is a list of items ordered by ranking score



---

Sequential rec & session-based rec

[blog](https://zhuanlan.zhihu.com/p/124324598) [blog](https://zhuanlan.zhihu.com/p/93965360)

session-based:

* 序列短。
* 匿名anonymous（不知道当前用户，所以难以做personalized rec）



Section 2.5 in the paper: in some SRS, one user-item interaction sequence includes multiple sub-sequences (also called sessions). 

In such a case, in addition to the prior interactions within the current subsequence, the historical sub-sequences may also influence the next user-item interaction to be predicted in the current subsequence [Ying et al., 2018 Sequential recommender system based on hierarchical attention networks. IJCAI]

Therefore, one more key challenge in SRSs is how to incorporate the hierarchical dependencies embedded in these two kinds of hierarchical structures into sequential dependency learning to generate more accurate sequential recommendations



Caser 是sequential rec

GRU4Rec, SR-GNN是session based



---

"long-term and high-order sequential dependencies."



Session-based recommendations with recurrent neural networks. ICLR2016

General factorization framework for context-aware recommendations 2016

Recurrent recommender networks ICDM2017

Towards neural mixture recommender for long range dependent user sequences WWW2019

---

"flexible order" (collective sequential dependencies)

​	RNN, Markov, factorization machines based, strict sequence order.

​	using CNN to model local and global dependencies



Personalized top-n sequential recommendation via convolutional sequence embedding. WSDM2018

​	Caser. sequential rec not session-based.

A simple convolutional generative network for next item recommendation.  WSDM2019

---

"learning seq dependencies attentively and discriminatively"

​	some interactions are irrelevant.



Attention based transactional context embedding for next-item recommendation AAAI2018

​	output a conditional prob

Sequential recommendation with user memory networks. ICDM2018

---

"heterogeneous relations"

​	mixture model integrates different types relations.

​	long (short) term seq dependency, similarity ...



Recommendation through mixtures of heterogeneous item relationships CIKM2018

Modeling multi-purpose sessions for next item recommendations via mixture-channel purpose routing network IJCAI2019

Towards neural mixture recommender for long range dependent user sequences WWW2019

---

"hierarchical structures"

​	meat-data (side info, features)

​	session. (one user-item interaction sequence includes multiple sub-sequence.)

​	

Personalizing session-based recommendations with hierarchical recurrent neural networks.  RecSys2017

Sequential recommender system based on hierarchical attention networks. IJCAI2018

---

traditional: sequence pattern mining & Markov-chain

Latent Representation: 

​	FM

​	embedding

​		Translation-based recommendation: A scalable method for modeling sequential behavior IJCAI2018

​			directly utilize embeddings to calculate a metric like the Euclidean distance as the interaction score. most use embeddings as input of NN.

RNN-based

CNN-based



GNN-based:

SR-GCN. AAAI2019 Session based recommendation with graph neural networks 

Weiping Song et al. - WSDM2019-based Social Recommendation via Dynamic Graph Attention Networks 

​		RNN 's hidden state as user interest.

​		graph: only need to consider one user, denoted as A.

​		node --->  interests of A and his friends

​		edge ---> A and A's friends

​		use GAT

Attention

Memory network:

​	Improving sequential recommendation with knowledge-enhanced memory networks SIGIR2018

​		output an interaction score

​	Sequential recommendation with user memory networks. ICDM2018

Mixture models





# Sequential RS

## Caser-WSDM18

Personalized top-n sequential recommendation via convolutional sequence embedding. WSDM2018

[code]( https://github.com/graytowne/caser_pytorch) sequential rec not session-based.

* general preferences. long term and static behaviors.
* sequential. recent and dynamic behaviors.

related but different work: temporal rec [14 31 34 26]

Section 1.2 in the paper, last paragraph.  about sequential pattern analysis.

model:  (Figure 3 in paper)

* For a user u, slide a window of size L + T over the user’s sequence, and each window generates a training instance
  for u, denoted by a triplet (u, previous L items, next T items)

* Embedding. L items embedding $E \in R^{L\times d}$. user embedding $p_u \in R^d$.
* CNN for $E$. horizontal and vertical. get sequential embedding $z$
* $p_u$ as general preferences. concate with $z$. as input of output layer.
* sample negative samples. train with "minimize negative samples likelihood, maximize positive samples likelihood"

experiments.

* dataset: MovieLens & Gowalla & Foursquare & Tmall. For a user sequence, the first 70% for train, the next 10% for validation, the last 20% for test.	
* metrices. Precision. Recall. MAP

虽然说是序列推荐，但在预测下一个item时，也只是用了最近的前k（k=5）个item，作为short-term preferences. static user embedding as long term preferences.



## SHAN-IJCAI18

Sequential recommender system based on hierarchical attention networks. IJCAI2018

users sequential interactions: $L_T = \{S_1, S_2, ..., S_T\}$.  At time step t, $S_t$ is a session, is a item set.

* short term preference. $S_t$
* long term preference. $L_{t-1}=S_1 \cup S_2 ... \cup S_{t-1}$

model in paper Figure 1

* use user and item id, and convert to embeddings.
* attention for long term preference. user embedding as query. item embeddings in $L_{t-1}$ as key and value.  get $u_{t-1}^{long}$
* attention for short term preference. user embedding as query. item embedding in $S_t$ and $u_{t-1}^{long}$ as key and value. get $u_t^{hybrid}$ as final user representation.

train with BPR loss & l2 reg. Metrics. Recall & AUC. Dataset: Tmall & Gowalla

Caser 提取short term preference with CNN. long term preference with a static user embedding. Though not using time info, but CNN is sensitive to item order in the sequence. But for SHAN, attention without time step information or item order info.



## MixtureModel-WWW19

Towards neural mixture recommender for long range dependent user sequences WWW2019

1) long range dependency analysis: Section 3.1 in paper.

2) Model: (Figure 2 in paper)

* embedding for each interaction in a given user seq. $\{item_1, item_2...item_t\}$

* three encoder built on Mixture-of-Experts (MOE) (Figure 3 in paper)
  * use the last interaction embedding
  * use RNN(lstm, gru) or CNN(TCN) to capture short range dependency. (RNN and TCN have comparable results)
  * use attention without position(time) info to capture long range dependency.
  * 输入输出维度合适时，除了RNN和CNN，其他不会引入额外参数
* use gate to aggregate the three encodings. and output layer with softmax.

* seq with sliding window and predict the $item_{t+1}$. and see Table 5 in paper, effect of three encoders.

3) When using RNN for sequence modeling, consider using attention or some simple rule to enhance sequence pattern.

（之前做KT时，在DKT基础上，用了最近的k个answer 和LSTM的输出一起来预测，发现AUC有0.5%~1%的提升。这里还是有疑惑的，这种简单的依赖关系，RNN应该很容易学到。但有时候简单的rule比attention效果相当或好一些。）



# Session based RS

## GRU4Rec

Session-based recommendations with recurrent neural networks. ICLR2016

 [GRU4Rec code](https://github.com/hungthanhpham94/GRU4REC-pytorch) [blog](https://zhuanlan.zhihu.com/p/28776432) session-based rec. (can be used for seqRS)

training strategy:

* batch-session training. 当该batch内的session结束时，用batch以外session填补，当然模型的参数要重置。

* negative sampling. (使用同batch的其他session的positive sample作为当前session的negative sampling)

loss:

* BRP
* TOP1 (BPR loss with regularization)

evaluation:

* recall@20 and MRR@20

exp conclusion: multi-layer GRU worse than one-layer GRU. one-hot better than embedding. tanh better. GRU better than RNN, LSTM.



## SR-GCN-AAAI19

AAAI2019 Session based recommendation with graph neural networks 

[code](https://github.com/CRIPAC-DIG/SR-GNN)

constructing graph

* node. items in the session. 
* edge. $(v_i, v_{i+1})$. item $v_{i+1}$ just after $v_i$ in session, then this two node have an edge.
* consider in-edge and out-edge. if no epeated items in the session, then just a chain graph. Figure 2 in paper. RNN just forward. SR-GCN consider in-edges and out-edges (bidirectional)

Node (item) embedding

* apply Gated-GNN on the graph. 

Session Embeddings (similar as graph pooling)

* current interests. use the last item's node embedding (WWW19 M3R mixture model中，有一个encoder也是使用序列中最后一个交互的embedding)
* long-term preference. attention.
* concate the above two vector

output layer

* output probability for all items using inner-dot. Eq (8)
* cross entropy loss.

experiment

* Yoochoose and Diginetica dataset. avg.length = 5~6  （session based rec 中session都不是很长，30那个量级。不像之前kt，序列长度200)
* Metrics. Precision@20 and MRR@20



