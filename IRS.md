# RL for IRS

* model the interactive recommendation process as a Markov Decision Process (MDP)

* model-based： Learning and adaptivity in interactive recommender systems 2007
* model-free methods
  * PG-based 
    * Haokun Chen et al. Large-scale interactive recommendation with tree-structured policy gradient. AAAI19
  * DQN based
    * Recommendations with negative feedback via pairwise deep reinforcement learning. In SIGKDD’18.
    * DRN: A deep reinforcement learning framework for news recommendation. www18.  "DQNR"
    * Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems.
  * DDPG based
    * Deep reinforcement learning in large discrete action spaces. 2015 "DDPG^{KNN}"
    * Reinforcement learning to rank in e-commerce search engine: Formalization, analysis, and application. In SIGKDD’18.
    * Deep reinforcement learning for page-wise recommendations. In ACM RecSys’18. "DDPGR"

***

## Env

RecoGym: A Reinforcement Learning Environment for the Problem of Product Recommendation in Online Advertising



## Large discrete action space

the calculation of output layer (softmax) is very large.

* Wolpertinger: map discrete action to continuous space, and then mapped to discrete valid action.

* TPGR: similar as "Hierarchical softmax".
* direct idea: refer to NLP, and use "hierarchical softmax" or negative sampling.
* but for Rec, usually choose a candidate item set for rec, not the whole item set. Thus, large action challenge may be not important.

***

Wolpertinger ($DDPG^{KNN}$)

Deep reinforcement learning in large discrete action spaces. 2015 DeepMind.

[code](https://github.com/nikhil3456/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces/blob/master/wolp.py) [blog](https://blog.csdn.net/qq_27465499/article/details/88185373) [blog](https://zhuanlan.zhihu.com/p/64514037) 

* discrete action space |A| is very big. represent the discrete actions in a continuous vectors

* Actor output proto-action in a continuous hidden space. $\pi(a/s) \in R^{d}$

* map to candidate discrete actions set. using approximate KNN (log-time).
* chooses the item with the max Q-value from the candidate items
* larger K boosts the performance but also brings computational cost



Action branching architectures for deep reinforcement learning. AAAI2018

​	such a method suffers from the inconsistency between the learned continuous action and the actually desired discrete action, and thereby may lead to unsatisfied results.

***

TPGR

Haokun Chen et al. Large-scale interactive recommendation with tree-structured policy gradient. In AAAI’19

using PG. (REINFORCE)

idea is similar as "Hierarchical softmax".

Further, study "negative sampling" and "noise contrastive estimation"

***

Youtube. Top-K Off-Policy Correction for a REINFORCE Recommender System，WSDM 2019

REINFORCEMENT Off policy



## DQN based

DRN & DEERS

* both define $(S, A, P, R, \gamma)$ and apply DQN. (DRN use dueling-double-DQN, DEERS just use DQN).
* DRN make change on reward (user click and user activeness) and exploration method (DBGD).
* DEERS make change on state (consider negative feedback) and loss (pairwise regularization).
* both methods make some changes and improve performance for real Rec scenarios based on DQN architecture.



***

Microsoft. DRN: A deep reinforcement learning framework for news recommendation. www18.

[blog](https://www.jianshu.com/p/4dfce7949fce)

Recommender system learns Q function to estimate the Q-value of all actions ("items")
at a given state ("user features or states"). then recommends item with highest Q-value at the current state.

* DQN - double - dueling	

  for dueling, calculate V(s) and A(s,a). input different features. state features for V(s,) |state features and actions features for A(s,a）

* reward is user click and user activeness (using survival analysis).

* exploration method. (previous $\epsilon$-greedy or UCB) 

  DBGD (dueling bandit gradient decent)

***

DEERS

[blog](https://zhuanlan.zhihu.com/p/58147362)

JD. KDD18 Recommendations with negative feedback via pairwise deep reinforcement learning 

* DQN
* state. contain positive items (user clicked or ordered) and negative items (user skipped). and modify Q network architecture. (details as original paper Figure 4)
* add pairwise regularization term. 

***



## DDPG based

DeepPage & LIRD

* both are JD's papers. Using DDPG with "Wolpertinger" strategy. DeepPage uses similarity to map output action to valid action. LIRD uses NN to generate score.

* LIRD builds a simulator for online training and testing.



***

DeepPage

JD. Recsys18 - Deep reinforcement learning for page-wise recommendations

DRL architecture selection:

* DQN with Q-net: input state, and output all Q(s,a).  can not handle large action space.
* DQN with Q-net: input <state, action>, output Q(s,a). high temporal complexity for large action space.
* Use actor-critic with deterministic policy (DDPG). input state, actor (Q-net) output a deterministic action. and critic make judgment.  But in this way, only can output continuous action space. Thus, apply  "Wolpertinger" strategy.



DDPG +  2-D page wise rec.

* state representation:

  encoder: state embedding - CNN (2-D page) - GRU - attention ---> get state repre at current time

* Actor: 

  decoder: state repre - deconvolution NN ---> generate proto-action(a page items emb) ---> valid items emb (most similar items with proto-action)

* critic:  Q(s,a)

* online training is same as Wolpertinger. (take action based on valid items. and calculate target-q-value and update actor based on actual output of actor.)  // online testing.

* offline training. paper Section 3.1.2  // offline test. paper Section 3.2.2

***

LIRD

JD. KDD19 - Deep Reinforcement Learning for List-wise Recommendations 

[code](https://github.com/luozachary/drl-rec) [blog](https://blog.csdn.net/a1066196847/article/details/104041757)

DRL architecture selection same as DeepPage

build simulator based on memory & similarity as env. 

* For offline training. given s, a. return reward.

* For offline evaluation. evaluate model using simulator. only have ground truth feedback of existing items in users' historical records, which are very sparse. 

* But simulator is not responsible for giving next state s'. (details as paper Algorithm 3)



***

DPG-FBE

Alibaba. Reinforcement learning to rank in e-commerce search engine: Formalization, analysis, and application. KDD’18.

learning to rank



## SIGIR2020-Paper

short paper: Reinforcement Learning based Recommendation with Graph Convolutional Q-network 

Very similar as KGQR.

* state: a sequence of observed items. (bot negative and positive feedback items)

* action: item

* reward: user give 1 or 0 feedback. (compared with KGQR, no sequence pattern).  

  “ In each T -step episode of user-agent interactions, the agent is forced to pick items from the available item set that consists of the 1000 sampled negative items and the observed positive items.” 

* DQN.

* for state and action, use user-item bipartite graph. Figure 1 and 2 in paper.

  similar as graphSage: for item i, sample fixed-sized 1-hop neighbors (users).

  similar as GAT. weighted aggregate neighbors info.

* use GRU for state seq with attention.	

---

KERL: A Knowledge-Guided Reinforcement Learning Model for Sequential Recommendation

Very similar as KGQR

* section 4.2 state representation

  state: interacted items sequence

  1) GRU for seq 2) KG with transE, and average item embedding in seq. 3) predicting future preference

* reward: 1) BLEU. make recommended item subseq similar as real item subseq. (exact match)  

  2) average item embedding in the rec and real subseq, and use con-similar.

没看懂，有时间参考代码[code](https://github.com/fanyubupt/KERL)

---



Jointly Non-Sampling Learning for Knowledge Graph Enhanced Recommendation



Adversarial Attack and Detection on Reinforcement Learning based Recommendation System

Leveraging Demonstrations for Reinforcement Recommendation Reasoning over Knowledge Graphs

MaHRL: Multi-goals Abstraction based Deep Hierarchical Reinforcement Learning for Recommendations

Self-Supervised Reinforcement Learning for Recommender Systems

Fairness-Aware Explainable Recommendation over Knowledge Graphs

