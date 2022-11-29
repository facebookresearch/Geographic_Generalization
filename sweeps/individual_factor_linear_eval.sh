
python evaluate.py -m model='glob(*)'\
	dataset=shapenet_individual_image_sampling\
	datamodule.factor_to_vary=Translation,Rotation,Scale,'Spot hue','Background path'\
	datamodule.train_prop_to_vary=0.0,0.05,0.25,0.5,0.75,0.95
