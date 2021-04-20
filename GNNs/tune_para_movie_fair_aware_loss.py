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
				for alpha in [1,10,20,30,40,50,60,70,80,90,100]: # alpha is lambda_{dp} or lambda_{eo} in the paper
					command = "python -m graphsaint.train_fair_aware_loss --gpu {} \
--outer_no {} --inner_no {} --loss_type {} --alpha {} --data_prefix {} \
--train_config {} --dataset {}".format(args.gpu,
						outer_no,inner_no, args.loss_type, alpha, 
						args.data_dir, args.train_config, dataset)
					print(command)
					os.system(command)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',type=str, default='MovieLens')
	parser.add_argument('--loss_type',type=str, default='dp',choices=['dp','eo'])  # dp,eo
	parser.add_argument('--train_config',type=str, default='train_config/ml_rw.yml') 
	parser.add_argument('--gpu',type=int, default=0)
	parser.add_argument('--num_folds_outer',type=int, default=3)
	parser.add_argument('--num_folds_inner',type=int, default=4)
	
	args = parser.parse_args()

	train(args)