### tf 分布式训练

https://www.tensorflow.org/extras/candidate_sampling.pdf

tf.feature_column https://zhuanlan.zhihu.com/p/73701872

[tf.train.SyncReplicasOptimizer](https://blog.csdn.net/qq_28626909/article/details/85003392): 异步梯度更新效果差。

​	match组，同步梯度更新，效率低。10个worker抵上1个local

​	[blog](https://markthink.gitbooks.io/caicloud/content/clever/develop/tutorials/core/sync-mode.html)

[分布式训练](https://blog.csdn.net/tiangcs/article/details/85952007)

https://zhuanlan.zhihu.com/p/56991108

[Parameter Server](https://zhuanlan.zhihu.com/p/82116922)

Session & Graph [zhihu](https://zhuanlan.zhihu.com/p/103996454) [zhihu](https://zhuanlan.zhihu.com/p/32869210)

tf.triain.MonitoredTrainingSession()， 会返回tf.train.MonitoredSession()的一个实例

​	MonitoredSession: Session-like object that handles initialization, recovery and hooks.

tf hook:  first, `MonitoredSession` should be used instead of normal `Session`.

[MonitoredSession&MonitoredTrainingSession&SessionRunHook](https://github.com/HustCoderHu/myNotes/blob/master/TensorFlow/train.MonitoredSession.md)

[MonitoredTrainingSession](https://blog.csdn.net/MrR1ght/article/details/81006343)

[SessionRunHook](https://blog.csdn.net/mrr1ght/article/details/81011280)

[hook](https://blog.csdn.net/yiqingyang2012/article/details/79917297) [hook执行顺序](https://www.jianshu.com/p/13734f112c43)

hook执行顺序：hooks都是继承父类tf.train.SessionRunHook()

```python
 """
  定义hook，只调用__init__()方法

	将定义好的hook实例，传入tf.MonitoredSession()中，此时会调用beigin()方法。

	session创建完，调用after_create_session()方法

  sess.run()前，会调用before_run()方法
  执行sess.run()
  sess.run()之后，会调用after_run()方法
	（这三个步骤是循环的，执行了几次sess.run(),上面三个步骤也会执行相应次数）

	sess.close() 或者 with ... sess:结束，会调用end()方法
 """
  call hooks.begin()
  sess = tf.Session()
  call hooks.after_create_session()
  while not stop is requested:  # py code: while not mon_sess.should_stop():
    call hooks.before_run()
    try:
      results = sess.run(merged_fetches, feed_dict=merged_feeds)
    except (errors.OutOfRangeError, StopIteration):
      break
    call hooks.after_run()
  call hooks.end()
  sess.close()
```



tf.estimator.Estimator

tf.feature_column https://blog.csdn.net/anshuai_aw1/article/details/105075335



xdl:

level1 ~ level4 https://yuque.antfin-inc.com/xdl-team/xdl2-design/pxadrv



https://yuque.antfin-inc.com/xdl-user/xdl2/nh9olu 

xdl的聚合函数

stage（对MonitoredTrainingSession的扩展吧） stage中可以使用hook

自定义stage：

​	函数：def my_stage(sess, ctx).  这种无法传入参数，

​	类：继承xdl.stage.Stage, 定义run()函数来执行。 可以实现传参

可以定义很多函数，但这些函数调用时，都要在with ctx.scope()下。



