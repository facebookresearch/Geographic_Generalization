
# python evaluate.py -m mode=local_test\
python evaluate.py -m model='glob(*)'\
	dataset=shapenet_all\
	datamodule.train_prop_to_vary=0.5
