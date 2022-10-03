cd tests/scripts && \
python preempt.py -m mode=local hydra/launcher=submitit_local\
	hydra/job_logging=colorlog\
	hydra/hydra_logging=colorlog\
	hydra.launcher.timeout_min=2\
	+hydra.launcher.gpus_per_node=1\
	+hydra.launcher.max_num_timeout=3\
	model=resnet_3d\
	trainer.max_epochs=100\ 
	+trainer.min_epochs=100\ 
	+trainer.limit_train_batches=5

# timeout_min will trigger requeueing after 2 min


# Uncomment to test on cluster
# cd tests && \
# python preempt.py -m mode=cluster hydra/launcher=submitit_slurm\
# 	hydra/job_logging=colorlog\
# 	hydra/hydra_logging=colorlog\
# 	hydra.launcher.timeout_min=4\
# 	hydra.launcher.max_num_timeout=3\
# 	hydra.launcher.partition=dev\
# 	trainer.gpus=1\
# 	trainer.max_epochs=100\ 
# 	trainer.limit_train_batches=5