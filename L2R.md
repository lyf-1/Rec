> The note is done when doing the click model task of dxy. 
>
> Need more work. 



# Model

RankNet [code](https://zhuanlan.zhihu.com/p/66497129): Christopher J.C. Burges. 2010. From RankNet to LambdaRank to LambdaMART: An overview. Technical Report. Microsoft Research  

ListNet [paper]( https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf) [pdf](https://oss.wanfangdata.com.cn/NewFulltext/Index?isread=true&type=perio&resourceId=qbxb201201008&transaction={"id"%3Anull%2C"transferOutAccountsStatus"%3Anull%2C"transaction"%3A{"id"%3A"1385837622139568128"%2C"status"%3A1%2C"createDateTime"%3Anull%2C"payDateTime"%3A1619244411393%2C"authToken"%3A"TGT-6865436-WcpdDHvCkiVAvMztn4CzXIdbRdybbbaKehi4bFeMFPghqdjOce-my.wanfangdata.com.cn"%2C"user"%3A{"accountType"%3A"Group"%2C"key"%3A"shjtdxip"}%2C"transferIn"%3A{"accountType"%3A"Income"%2C"key"%3A"PeriodicalFulltext"}%2C"transferOut"%3A{"GTimeLimit.shjtdxip"%3A3.0}%2C"turnover"%3A3.0%2C"orderTurnover"%3A3.0%2C"productDetail"%3A"perio_qbxb201201008"%2C"productTitle"%3Anull%2C"userIP"%3A"111.186.3.161"%2C"organName"%3Anull%2C"memo"%3Anull%2C"orderUser"%3A"shjtdxip"%2C"orderChannel"%3A"pc"%2C"payTag"%3A""%2C"webTransactionRequest"%3Anull%2C"signature"%3A"gYjpX1ArKm2KObtldCMPqotOIQ7iv%2FE3KLgGMUHm%2BGZlhYLsigPAgKri96DBU69e8IXvwBXmQ92O\n7q2CLVUqN8O%2BAep%2Bn4cdUz%2BacphJvDfO2yh2gyetlq5cnDBywZHTH7NVlh8L5IPGeE06bF6cKVQR\nIGN1kz6hNbQNqj29RNE%3D"%2C"delete"%3Afalse}%2C"isCache"%3Afalse}) 

DBGD (online) [paper](https://www.cs.cornell.edu/people/tj/publications/yue_joachims_09a.pdf) [blog](https://www.cnblogs.com/bentuwuying/p/6690836.html)  

PDGD: Differentiable Unbiased Online Learning to Rank 



# Metrics

NDCG: 衡量ranker的好坏的。给定一个ranking list，其NDCG值就确定了。对于相同一组doc，不同ranker给出不同的排序方式，用NDCG来衡量哪种排序更好。（计算NDCG是需要有标注的relevance label的） 

Click model 的衡量通常有两类task. 给定一个query，以及一个已经排序的doc list，以及对应的click list.

* 第一个任务是Click prediction，衡量指标有LL， PPL.  *
* 第二个任务是relevance estimation。对于传统的PGM的click model，relevance可以定义为Attractiveness 以及satisfaction等，然后position bias在examination 假设中包含。这样，预测的relevance就是less biased的。然后基于预测的relevance，由大到小重新对doc list进行排序，然后计算NDCG值。以及衡量relevance 估计的准确与否。对于NCM，CACM，AICM这些模型，为了得到unbiased 的relevance估计，可以通过把每个doc当做一个序列，来预测点击概率，作为relevance的估计。（对于TianGong数据集，relevance label本身就是与位置有关的，本身就是biased的，所以预测relevance时，在原始顺序的doc list上，序列预测每一个doc的点击概率，当作q-d pair的relevance）  

