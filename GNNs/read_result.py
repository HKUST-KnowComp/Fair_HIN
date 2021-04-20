import numpy as np 
from os.path import join, exists
import argparse
import json


def get_mean_std_gnn_base(args):

	dev_mrr_results = []
	test_mrr_results = []
	dev_fair_results = []
	test_fair_results = []
	
	with open(join(args.data_dir,'{}_{}_{}.txt'.format(args.method,args.criterion,args.dataset))) as fin:
		for line in fin:
			if line[:len('gnn_base')] == 'gnn_base':
				_, _, _, _, _ = line.strip().split()
			elif line[:len('dev_mrr')] == 'dev_mrr':
				_, dev_mrr, _, dev_fair = line.strip().split()
				dev_mrr = float(dev_mrr)
				dev_fair = float(dev_fair)
				dev_mrr_results.append(dev_mrr)	
				dev_fair_results.append(dev_fair)

			elif line[:len('test_mrr')] == 'test_mrr':
				_, test_mrr, _, test_fair = line.strip().split()
				test_mrr = float(test_mrr)
				test_fair = float(test_fair)
				test_mrr_results.append(test_mrr)	
				test_fair_results.append(test_fair)
		
	print("dev mrr  {:1.5f}".format( np.mean(dev_mrr_results) ))
	print("dev {}   {:1.5f}".format(args.criterion, np.mean(dev_fair_results) ))
	print("test mrr {:1.5f}".format( np.mean( test_mrr_results) ))
	print("test {}  {:1.5f}".format(args.criterion, np.mean( test_fair_results) ))

def get_mean_std_adv(args):
	if args.dataset == 'ml':
		with open(args.thres_file, 'r') as fin:
			threshold = json.load(fin)
		# threshold = {}
		# threshold['dp'] = {'dev_low':0.1084,'dev_med':0.1626,'dev_high':0.2168,\
		# 				'test_low':0.0992,'test_med':0.1488,'test_high':0.1984}
		# threshold['eo'] = {'dev_low':0.0590,'dev_med':0.0884,'dev_high':0.1179,\
		# 				'test_low':0.0524,'test_med':0.0786,'test_high':0.1047}	

	dev_mrr_results = {}
	test_mrr_results = {}
	dev_fair_results = {}
	test_fair_results = {}
	discard = False

	with open(join(args.data_dir,'{}_{}_{}_{}.txt'.format(args.method,args.criterion,args.fair_level,args.dataset))) as fin:
		for line in fin:
			if line[:len('gamma')] == 'gamma':
				_, gamma, _, _, _, _ = line.strip().split()
			elif line[:len('dev_mrr')] == 'dev_mrr':
				_, dev_mrr, _, dev_fair = line.strip().split()
				dev_mrr = float(dev_mrr)
				dev_fair = float(dev_fair)
				if dev_fair < threshold[args.criterion]['dev_'+args.fair_level]:
					if gamma not in dev_mrr_results:
						dev_mrr_results[gamma] = [] 
					dev_mrr_results[gamma].append(dev_mrr)
					if gamma not in dev_fair_results:
						dev_fair_results[gamma] = [] 
					dev_fair_results[gamma].append(dev_fair)
				else:
					# print('discard',line)
					discard = True
			elif line[:len('test_mrr')] == 'test_mrr':
				_, test_mrr, _, test_fair = line.strip().split()
				test_mrr = float(test_mrr)
				test_fair = float(test_fair)
				if not discard:
					if gamma not in test_mrr_results:
						test_mrr_results[gamma] = [] 
					test_mrr_results[gamma].append(test_mrr)
					if gamma not in test_fair_results:
						test_fair_results[gamma] = [] 
					test_fair_results[gamma].append(test_fair)
				else:
					# print('discard',line)
					pass
				discard = False
	dev_list = [(gamma,np.mean(dev_mrr_results[gamma])) for gamma in dev_mrr_results]
	test_list = [(gamma,np.mean(test_mrr_results[gamma])) for gamma in test_mrr_results]

	dev_list= sorted(dev_list,key=lambda tup:tup[1])
	best_gamma, best_dev_mrr = dev_list[-1]
	
	assert best_dev_mrr == np.mean( dev_mrr_results[best_gamma])
	print("dev mrr  {:1.5f}".format( best_dev_mrr ))
	print("dev {}   {:1.5f}".format(args.criterion, np.mean(dev_fair_results[best_gamma]) ))
	print("test mrr {:1.5f}".format( np.mean( test_mrr_results[best_gamma]) ))
	print("test {}  {:1.5f}".format(args.criterion, np.mean( test_fair_results[best_gamma]) ))
	

def get_mean_std_fair_loss(args):
	if args.dataset == 'ml':
		with open(args.thres_file, 'r') as fin:
			threshold = json.load(fin)
		# threshold = {}
		# threshold['dp'] = {'dev_low':0.1084,'dev_med':0.1626,'dev_high':0.2168,\
		# 				'test_low':0.0992,'test_med':0.1488,'test_high':0.1984}
		# threshold['eo'] = {'dev_low':0.0590,'dev_med':0.0884,'dev_high':0.1179,\
		# 				'test_low':0.0524,'test_med':0.0786,'test_high':0.1047}
		

	dev_mrr_results = {}
	test_mrr_results = {}
	dev_fair_results = {}
	test_fair_results = {}
	discard = False
	with open(join(args.data_dir,'{}_{}_{}_{}.txt'.format(args.method,args.criterion,args.fair_level,args.dataset))) as fin:
		for line in fin:
			if line[:len('alpha')] == 'alpha':
				_, alpha, _, _, _, _ = line.strip().split()
			elif line[:len('dev_mrr')] == 'dev_mrr':
				_, dev_mrr, _, dev_fair = line.strip().split()
				dev_mrr = float(dev_mrr)
				dev_fair = float(dev_fair)
				if dev_fair < threshold[args.criterion]['dev_'+args.fair_level]:
					if alpha not in dev_mrr_results:
						dev_mrr_results[alpha] = [] 
					dev_mrr_results[alpha].append(dev_mrr)
					if alpha not in dev_fair_results:
						dev_fair_results[alpha] = [] 
					dev_fair_results[alpha].append(dev_fair)
				else:
					# print('discard',line)
					discard = True
			elif line[:len('test_mrr')] == 'test_mrr':
				_, test_mrr, _, test_fair = line.strip().split()
				test_mrr = float(test_mrr)
				test_fair = float(test_fair)
				if not discard:
					if alpha not in test_mrr_results:
						test_mrr_results[alpha] = [] 
					test_mrr_results[alpha].append(test_mrr)
					if alpha not in test_fair_results:
						test_fair_results[alpha] = [] 
					test_fair_results[alpha].append(test_fair)
				else:
					# print('discard',line)
					pass
				discard = False
	dev_list = [(alpha,np.mean(dev_mrr_results[alpha])) for alpha in dev_mrr_results]
	test_list = [(alpha,np.mean(test_mrr_results[alpha])) for alpha in test_mrr_results]
	
	dev_list= sorted(dev_list,key=lambda tup:tup[1])
	best_alpha, best_dev_mrr = dev_list[-1]
	
	assert best_dev_mrr == np.mean( dev_mrr_results[best_alpha])
	print("dev mrr  {:1.5f}".format( best_dev_mrr ))
	print("dev {}   {:1.5f}".format(args.criterion, np.mean(dev_fair_results[best_alpha]) ))
	print("test mrr {:1.5f}".format( np.mean( test_mrr_results[best_alpha]) ))
	print("test {}  {:1.5f}".format(args.criterion, np.mean( test_fair_results[best_alpha]) ))
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='output_result')
	parser.add_argument('--method', type=str, default='fair_loss',choices=['adv','fair_loss','gnn_base'])
	parser.add_argument('--dataset', type=str, default="ml")
	parser.add_argument('--criterion', type=str, default="eo")
	parser.add_argument('--fair_level', type=str, default="high")
	parser.add_argument('--thres_file', type=str, default="thres_fair_conditions.json")
	args = parser.parse_args()

	if args.method == 'fair_loss':
		get_mean_std_fair_loss(args)
	elif args.method == 'adv':
		get_mean_std_adv(args)
	elif args.method == 'gnn_base':
		get_mean_std_gnn_base(args)


