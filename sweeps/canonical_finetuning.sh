# learning rates: np.logspace(-2,-6, num=6)

# python evaluate.py -m mode=cluster model=vicreg_pretrained_1k,vicreg_pretrained_21k\
python evaluate.py -m model='glob(*)'\
	dataset=shapenet_canonical\
	evaluation=finetuning\
	evaluation_module.learning_rate=1.00e-02,1.58e-03,2.51e-04,3.98e-05,6.31e-06,1.00e-06