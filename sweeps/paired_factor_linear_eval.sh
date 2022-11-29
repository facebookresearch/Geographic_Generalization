 # Translation,Rotation,Scale,'Spot hue','Background path'

# python evaluate.py -m mode=local_test\
python evaluate.py -m model='glob(*)'\
	dataset=shapenet_pairs\
	datamodule.factors_to_vary='[Translation, Rotation]','[Translation, Scale]','[Translation, Spot hue]','[Translation, Background path]','[Rotation, Scale]','[Rotation, Spot hue]','[Rotation, Background path]','[Scale, Spot hue]','[Scale, Background path]','[Spot hue, Background path]'\
	datamodule.train_prop_to_vary=0.0,0.5
