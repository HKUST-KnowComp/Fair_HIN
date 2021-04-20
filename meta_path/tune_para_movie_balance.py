import os
import numpy as np
import argparse
import math
from os.path import join, exists

def train(num_walks,walk_len):
	result_file = 'res_w{}.l{}.cupuc.upu.w{}.l{}.balance.txt'.format(num_walks,walk_len,num_walks,walk_len)
	result_file = os.path.join(args.result_dir, result_file)
	emb_file = "w{}.l{}.cupuc.upu.w{}.l{}.balance.txt".format(num_walks,walk_len,num_walks,walk_len)

	if not os.path.exists(os.path.join(args.emb_dir,emb_file)): 
		command = "python py4genMetaPaths_movie.py \
--walks {} --length {} --upu_walks {} --upu_length {} \
--output_data_dir {} --emb_dir {} \
--data_dir {} --c --balance".format(
				num_walks, walk_len, num_walks ,walk_len, \
				args.output_data_dir, args.emb_dir, \
				args.data_dir)
		print(command)
		os.system(command)


	command = "\nrm {}".format(result_file)
	print(command)
	os.system(command)

	for outer_no in range(args.num_folds_outer):
		for inner_no in range(args.num_folds_inner):
			for k in range(5):
				command = "CUDA_VISIBLE_DEVICES={} python eval_link_prediction_movie_balance_fair_constraint.py --emb {} \
		--emb_dir {} --data_dir {} --num_folds_inner {} --inner_no {} --outer_no {} --method m2v_balance >> {}".format(
					args.gpu,
					emb_file,
					args.emb_dir,
					args.data_dir,
					args.num_folds_inner,
					inner_no,
					outer_no,
					result_file)
				print(command)
				os.system(command)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--output_data_dir',type=str, default="path")  #/data/zzengae/metapath2vec/path/movie/
	parser.add_argument('--emb_dir',type=str, default="emb")  # /data/zzengae/code_metapath2vec/emb/movie

	parser.add_argument('--data_dir',type=str, default='MovieLens')  # cpu4
	parser.add_argument('--result_dir',type=str, default='result_movie_balance') 

	parser.add_argument('--gpu',type=int, default=0)
	parser.add_argument('--num_folds_outer',type=int, default=3)
	parser.add_argument('--num_folds_inner',type=int, default=4)
	
	args = parser.parse_args()
	
	if not os.path.exists(args.output_data_dir):
		os.makedirs(args.output_data_dir)

	if not os.path.exists(args.emb_dir):
		os.makedirs(args.emb_dir)

	if not os.path.exists(args.result_dir):
		os.makedirs(args.result_dir)
	
	for num_walks in [50,100,150,200]:
		for walk_len in [50,100,150,200]:
			train(num_walks,walk_len)

