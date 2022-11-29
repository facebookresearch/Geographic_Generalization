
python store_predictions.py -m model='glob(*)'\
	evaluation_module.eval_type='linear_eval','finetuning'\
	hydra.launcher.timeout_min=120

