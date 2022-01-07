source ~/envs/anaconda3/bin/activate tf1.x
export CUDA_VISIBLE_DEVICES=1

#with/withour BN, with large/small weights.
#export SETTING_ARGS="--ubn --save_path ./rst/BNS"
#export SETTING_ARGS="--ubn --ulw --save_path ./rst/BNL"
#export SETTING_ARGS="--save_path ./rst/NBNS"
#export SETTING_ARGS="--ulw --save_path ./rst/NBNL"

#BN without tf.control_dependencies(). moving_mean|moving_variance is not updated.
#export SETTING_ARGS="--ubn --save_path ./rst/BNS_noctrl"
export SETTING_ARGS="--ubn --ulw --save_path ./rst/BNL_noctrl"
#export SETTING_ARGS="--ubn --save_path ./rst/test --epochs 1 --bs 256"

python main.py $SETTING_ARGS
