
python evaluate.py -m model='glob(*)'\
	+experiment=random_class_generalization\
	datamodule.factor_to_vary=Translation,Rotation,Scale,'Spot hue','Background path'\
	datamodule.train_prop_to_vary=0.5
