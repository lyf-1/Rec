# Source Code

* data.py: get mnist dataset.

  ```python
  mnist = tf.keras.datasets.mnist # minist data is saved in the folder ~/.keras
  mnist.load_data()               # train | valid | test dataset.
  ```

* model.py: define FNN-with/without-BatchNormalization model for mnist. (tf.layers.batch_normalization()) 

* main.py: train/test/hyper-params. | plot_exp_results.


# Exp Settings
* BN (--ubn) 
  * large weight (--ulw)    [BNL]
  * small weight               [BNS]

* No BN
  * large weight (--ulw)    [NBNL]
  * small weight               [NBNS]


* tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    bn/moving_mean|moving_variance: zero-init|one-init
    without: moving_mean|moving_variance is not updated. still zero|one. (comment out the code in model.py) [BNL_noctrl | BNS_noctrl]
    with: moving_mean|moving_variance is updated.



# Exp Results
|          | BNS    | BNL    | NBNS   | NBNL   | BNS_noctrl | BNL_noctrl |
| -------- | ------ | ------ | ------ | ------ | ---------- | ---------- |
| Test ACC | 0.9816 | 0.9500 | 0.9760 | 0.1140 | 0.8045     | 0.5468     |

./rst/learning_curve.pdf show the training curve of the above settings.



# TODO
1)
init = tf.truncated_normal_initializer(mean=10, stddev=0.1)
init = tf.truncated_normal_initializer(mean=0, stddev=0.1)
2)
init = tf.truncated_normal_initializer(mean=0, stddev=10)
init = tf.truncated_normal_initializer(mean=0, stddev=0.1)

1) has bad training curve. training loss does not decrease.

