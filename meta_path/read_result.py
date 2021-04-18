import numpy as np 
from os.path import join, exists
import argparse

def read(args):
	result = {}
	with open(join(args.data_dir,
		"{}_{}_{}_{}.txt".format(
			args.method, args.criterion, args.fair_level, args.dataset))) as fin:
		for line in fin:
			line = line.strip().split()
			if len(line) == 6:
				_, para, _, outer_no, _, inner_no = line
				if para not in result:
					result[para] = {}
			elif len(line) == 4:
				field_name_0, field_value_0, field_name_1, field_value_1 = line
				if field_name_0 not in result[para].keys():
					result[para][field_name_0] = []
				if field_name_1 not in result[para].keys():
					result[para][field_name_1] = []
				result[para][field_name_0].append(float(field_value_0)) 
				result[para][field_name_1].append(float(field_value_1))

		to_be_sort = []
		for para in result:
			to_be_sort.append((para,np.mean(result[para]['dev_mrr'])))
		to_be_sort = sorted(to_be_sort,key=lambda elem:elem[1])
		best_para, best_dev_mrr = to_be_sort[-1]
		
		print("dev mrr {:1.5f}".format(np.mean(result[best_para]['dev_mrr'])))
		print("dev {} {:1.5f}".format(args.criterion, np.mean(result[best_para]['dev_{}'.format(args.criterion)])))
		print("test mrr {:1.5f}".format(np.mean(result[best_para]['test_mrr'])))
		print("test {} {:1.5f}".format(args.criterion, np.mean(result[best_para]['test_{}'.format(args.criterion)])))
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='.')
	parser.add_argument('--method', type=str, default='m2v_bias')
	parser.add_argument('--criterion', type=str, default="eo")
	parser.add_argument('--fair_level', type=str, default="high")
	parser.add_argument('--dataset', type=str, default="ml")
	args = parser.parse_args()
	read(args)