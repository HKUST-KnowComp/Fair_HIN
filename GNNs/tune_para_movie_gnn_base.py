import os
import numpy as np
import argparse
import math
from os.path import join, exists

def train(args):
	if args.data_dir == "MovieLens":
		dataset = "ml"
	for i in range(5):
		for outer_no in range(args.num_folds_outer):
			for inner_no in range(args.num_folds_inner):
					command = "python -m graphsaint.train_bl --fair False --gpu {} \
--outer_no {} --inner_no {}  --data_prefix {} \
--train_config {} --dataset {} ".format(args.gpu,
						outer_no,inner_no, 
						args.data_dir, args.train_config, dataset)
					print(command)
					os.system(command)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',type=str, default='MovieLens')
	parser.add_argument('--train_config',type=str, default='train_config/ml_rw_gnn_base.yml') 
	parser.add_argument('--gpu',type=int, default=0)
	parser.add_argument('--num_folds_outer',type=int, default=3)
	parser.add_argument('--num_folds_inner',type=int, default=4)
	
	args = parser.parse_args()

	train(args)