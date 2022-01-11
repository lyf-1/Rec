#source ~/envs/anaconda3/bin/activate tf1.x
source ~/miniconda3/bin/activate py27
export CUDA_VISIBLE_DEVICES=1

#without BN and LN, with large/small weights.
#export SETTING_ARGS="--save_path ./rst/NBNS"
#export SETTING_ARGS="--ulw --save_path ./rst/NBNL"

#with BN, with large/small weights.
#export SETTING_ARGS="--ubn --save_path ./rst/BNS"
#export SETTING_ARGS="--ubn --ulw --save_path ./rst/BNL"

#with LN, with large/small weights.
export SETTING_ARGS="--uln --save_path ./rst/LNS"
#export SETTING_ARGS="--uln --ulw --save_path ./rst/LNL"

#BN without tf.control_dependencies(). moving_mean|moving_variance is not updated.
#export SETTING_ARGS="--ubn --save_path ./rst/BNS_noctrl"
#export SETTING_ARGS="--ubn --ulw --save_path ./rst/BNL_noctrl"
#export SETTING_ARGS="--ubn --save_path ./rst/test --epochs 1 --bs 256"

python main.py $SETTING_ARGS
