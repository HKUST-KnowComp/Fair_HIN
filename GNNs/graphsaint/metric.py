from sklearn import metrics
import numpy as np
import tensorflow as tf
from graphsaint.globals import *
import json
import os
import math
import copy
from collections import Counter

def calc_f1(y_true, y_pred): 
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    return metrics.f1_score(y_true, y_pred, average="weighted")

def calc_acc(y_true, y_pred): 
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    return metrics.accuracy_score(y_true, y_pred)

def mean_reciprocal_rank(y_true, y_pred):
	y_true = copy.deepcopy(y_true)
	y_pred = copy.deepcopy(y_pred)	

	y_true = np.argmax(y_true, axis=1)

	mrr = 0.0
	for i in range(len(y_true)):
		index = np.argsort(y_pred[i])[::-1]
		mrr += 1/(index.tolist().index(y_true[i])+1)
	return mrr / len(y_true)

def fairness(y_true,y_pred,users,name):

	y_true = np.argmax(y_true, axis=1)
	y_pred = np.argmax(y_pred, axis=1)

	conc_to_lable = json.load(open(os.path.join(FLAGS.data_prefix,'conc_to_lable.json'),'r'))
	conc_label_to_name = json.load(open(os.path.join(FLAGS.data_prefix,'conc_label_to_name.json'),'r'))

	conc_label_to_name = { int(k):conc_label_to_name[k] for k in conc_label_to_name.keys() }

	# users = json.load(open(os.path.join(FLAGS.data_prefix,name+'_users.json'),'r'))

	male_users = json.load(open(os.path.join(FLAGS.data_prefix,'male_users.json'),'r'))
	female_users = json.load(open(os.path.join(FLAGS.data_prefix,'female_users.json'),'r'))

	common = set(users) & set(male_users) 
	print('male',len(common))
	common = set(users) & set(female_users)
	print('female',len(common))
	
	print('='*40)
	
	results_by_gender= { 'male':[], 'female':[] }
	labels_by_gender= { 'male':[], 'female':[] }
	
	for r, l, u in zip(y_pred,y_true,users):
		if u in male_users:
			results_by_gender['male'].append(r)
			labels_by_gender['male'].append(l)
		elif u in female_users:
			results_by_gender['female'].append(r)
			labels_by_gender['female'].append(l)

	male_dist = np.asarray([0.0 for _ in range(len(conc_to_lable))])
	female_dist = np.asarray([0.0 for _ in range(len(conc_to_lable))])

	# demog_disparity 

	predicted = results_by_gender['male']
	for p in predicted:
		male_dist[p] += 1

	predicted = results_by_gender['female']
	for p in predicted:
		female_dist[p] += 1

	norm_male_dist = male_dist/sum(male_dist)
	norm_female_dist = female_dist/sum(female_dist)
	
	smooth_norm_male_dist = (male_dist+1)/(sum(male_dist)+len(conc_to_lable))
	smooth_norm_female_dist = (female_dist+1)/(sum(female_dist)+len(conc_to_lable))
	
	gender_demog_parity =  0.5 * np.sum(np.abs(norm_male_dist-norm_female_dist))


	print('{:25} {:.4f}'.format(name+'_gender_demog_parity', gender_demog_parity))
	print("{:25} {:13} {:13}".format('concentration','male','female'))
	for idx in np.argsort(np.abs(norm_male_dist-norm_female_dist))[::-1]:
		print("{:25} {:.4f} {:.4f} {:.4f} {:.4f}".format(conc_label_to_name[idx],
			norm_male_dist[idx],smooth_norm_male_dist[idx],
			norm_female_dist[idx],smooth_norm_female_dist[idx]))
	print('end_of_{}_gender_demog_parity'.format(name))
	print('='*40)


	# equal odds
	groups = [[] for _ in range(len(conc_to_lable))]
	for r, l in zip(results_by_gender['male'],labels_by_gender['male']):
		groups[l].append(r)

	confusion_matrix_male = np.zeros([len(conc_to_lable),len(conc_to_lable)])
	for l in range(len(conc_to_lable)):
		if len(groups[l]) == 0:
			continue
		predicted = groups[l]
		for p in predicted:
			confusion_matrix_male[l][p] += 1
	sum_ = confusion_matrix_male.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_male_norm = confusion_matrix_male / sum_

	groups = [[] for _ in range(len(conc_to_lable))]
	for r, l in zip(results_by_gender['female'],labels_by_gender['female']):
		groups[l].append(r)
	confusion_matrix_female = np.zeros([len(conc_to_lable),len(conc_to_lable)])
	for l in range(len(conc_to_lable)):
		if len(groups[l]) == 0:
			continue
		predicted = groups[l]
		for p in predicted:
			confusion_matrix_female[l][p] += 1
	sum_ = confusion_matrix_female.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_female_norm = confusion_matrix_female / sum_

	gender_confusion = np.abs(confusion_matrix_male_norm - confusion_matrix_female_norm)

	diag = gender_confusion.diagonal()
	total = 0
	num = 0
	print('{}_gender_equal_odds'.format(name))
	for i in np.argsort(diag)[::-1]:
		sum_1 = np.sum(confusion_matrix_male[i])
		sum_2 = np.sum(confusion_matrix_female[i])
		if confusion_matrix_male[i][i] == 0 and confusion_matrix_female[i][i] == 0:
			continue
		print("{:25} {:.4f}({:.1f}/{:.1f}) {:.4f}({:.1f}/{:.1f})".format(conc_label_to_name[i],
			confusion_matrix_male_norm[i][i],
			confusion_matrix_male[i][i],
			np.sum(confusion_matrix_male[i]),
			confusion_matrix_female_norm[i][i],
			confusion_matrix_female[i][i],
			np.sum(confusion_matrix_female[i])))
		total += np.abs(confusion_matrix_male_norm[i][i]-confusion_matrix_female_norm[i][i])
		num += 1
	print('end_of_{}_gender_equal_odds'.format(name))
	print('='*40)
	print("{:25} {:.4f} {:.4f}".format(name+'_difference',total,total/num))
	print('='*40)

def read_result(file_name,name):
	demog_parity = []
	equal_odd_avg = []
	equal_odd_total = []
	demog_parity_smooth_epsilon_max = []
	demog_parity_smooth_epsilon_avg = []
	equal_odd_start_flag = False
	demog_parity_start_flag = False
	
	with open(file_name) as fin:
		for line in fin:
			if line[:len(name+'_gender_demog_parity')] == name+'_gender_demog_parity':
				demog_parity.append(float(line.strip().split()[-1]))
				demog_parity_start_flag = True
				demog_parity_smooth_epsilon_each_conc = []
				
			elif line[:len(name+'_gender_equal_odds')] == name+'_gender_equal_odds':
				equal_odd_start_flag = True
				total_non_zero = 0
				equal_odd_each_conc = []

			elif line[:len(name+'_difference')] == name+'_difference':
				equal_odd_total_each_time = float(line.strip().split()[-2])

			elif line[:len("end_of_{}_gender_demog_parity".format(name))] == "end_of_{}_gender_demog_parity".format(name):
				demog_parity_smooth_epsilon_max.append(np.max(demog_parity_smooth_epsilon_each_conc))
				demog_parity_smooth_epsilon_avg.append(np.mean(demog_parity_smooth_epsilon_each_conc))
				demog_parity_start_flag = False

			elif line[:len("end_of_{}_gender_equal_odds".format(name))] == "end_of_{}_gender_equal_odds".format(name):
				equal_odd_start_flag = False
				equal_odd_total.append(np.sum(equal_odd_each_conc))
				equal_odd_avg.append(np.mean(equal_odd_each_conc))

			elif equal_odd_start_flag: # architecture 0.5000(2.0/4.0) 0.0000(0.0/6.0)
				female_acc = float(line.strip().split()[-1].split('(')[0])
				male_acc = float(line.strip().split()[-2].split('(')[0])
				if female_acc != 0.0 or male_acc != 0.0:
					equal_odd_each_conc.append(abs(female_acc-male_acc))

			elif demog_parity_start_flag: # architecture 0.5000(2.0/4.0) 0.0000(0.0/6.0)
				if line.strip().split() == ['concentration', 'male', 'female']:
					continue
				else:
					female_acc = float(line.strip().split()[-1])
					male_acc = float(line.strip().split()[-3])
					if female_acc != 0.0 and male_acc != 0.0:
						smooth_e = max(math.log(female_acc/male_acc),math.log(male_acc/female_acc))
						demog_parity_smooth_epsilon_each_conc.append(smooth_e)

			

	mean_demog_parity = np.mean(demog_parity)
	mean_equal_odd_avg = np.mean(equal_odd_avg)
	mean_equal_odd_total = np.mean(equal_odd_total)
	mean_demog_parity_smooth_epsilon_max = np.mean(demog_parity_smooth_epsilon_max)
	mean_demog_parity_smooth_epsilon_avg = np.mean(demog_parity_smooth_epsilon_avg)
	
	
	return mean_demog_parity, mean_equal_odd_avg, mean_equal_odd_total, mean_demog_parity_smooth_epsilon_max, mean_demog_parity_smooth_epsilon_avg

def fairness_silent(y_true,y_pred,users,name,loss_type):

	import time
	result_file = 'res_{}.txt'.format(time.time())

	fo = open(result_file,'w')

	y_true = np.argmax(y_true, axis=1)
	y_pred = np.argmax(y_pred, axis=1)

	conc_to_lable = json.load(open(os.path.join(FLAGS.data_prefix,'conc_to_lable.json'),'r'))
	conc_label_to_name = json.load(open(os.path.join(FLAGS.data_prefix,'conc_label_to_name.json'),'r'))

	conc_label_to_name = { int(k):conc_label_to_name[k] for k in conc_label_to_name.keys() }

	male_users = json.load(open(os.path.join(FLAGS.data_prefix,'male_users.json'),'r'))
	female_users = json.load(open(os.path.join(FLAGS.data_prefix,'female_users.json'),'r'))

	common = set(users) & set(male_users) 
	fo.write('male ' + str(len(common)) + '\n')
	
	common = set(users) & set(female_users)
	fo.write('female ' + str(len(common)) + '\n')
	
	fo.write('='*40+'\n')
	
	results_by_gender= { 'male':[], 'female':[] }
	labels_by_gender= { 'male':[], 'female':[] }
	
	for r, l, u in zip(y_pred,y_true,users):
		if u in male_users:
			results_by_gender['male'].append(r)
			labels_by_gender['male'].append(l)
		elif u in female_users:
			results_by_gender['female'].append(r)
			labels_by_gender['female'].append(l)

	male_dist = np.asarray([0.0 for _ in range(len(conc_to_lable))])
	female_dist = np.asarray([0.0 for _ in range(len(conc_to_lable))])

	# demog_disparity 

	predicted = results_by_gender['male']
	for p in predicted:
		male_dist[p] += 1

	predicted = results_by_gender['female']
	for p in predicted:
		female_dist[p] += 1


	# fo.write('zzq sum(male_dist) ' + str(sum(male_dist)) + '\n')
	# fo.write('zzq sum(female_dist) ' + str(sum(female_dist)) + '\n')

	norm_male_dist = male_dist/sum(male_dist)
	norm_female_dist = female_dist/sum(female_dist)
	
	smooth_norm_male_dist = (male_dist+1)/(sum(male_dist)+len(conc_to_lable))
	smooth_norm_female_dist = (female_dist+1)/(sum(female_dist)+len(conc_to_lable))
	
	gender_demog_parity =  0.5 * np.sum(np.abs(norm_male_dist-norm_female_dist))


	# print('{:25} {:.4f}'.format(name+'_gender_demog_parity', gender_demog_parity))
	# print("{:25} {:13} {:13}".format('concentration','male','female'))
	# for idx in np.argsort(np.abs(norm_male_dist-norm_female_dist))[::-1]:
	# 	print("{:25} {:.4f} {:.4f} {:.4f} {:.4f}".format(conc_label_to_name[idx],
	# 		norm_male_dist[idx],smooth_norm_male_dist[idx],
	# 		norm_female_dist[idx],smooth_norm_female_dist[idx]))
	# print('end_of_{}_gender_demog_parity'.format(name))
	# print('='*40)


	fo.write('{:25} {:.4f}\n'.format(name+'_gender_demog_parity', gender_demog_parity))
	fo.write("{:25} {:13} {:13}\n".format('concentration','male','female'))
	for idx in np.argsort(np.abs(norm_male_dist-norm_female_dist))[::-1]:
		fo.write("{:25} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(conc_label_to_name[idx],
			norm_male_dist[idx],smooth_norm_male_dist[idx],
			norm_female_dist[idx],smooth_norm_female_dist[idx]))
	fo.write('end_of_{}_gender_demog_parity\n'.format(name))
	fo.write('='*40+'\n')

	# equal odds
	groups = [[] for _ in range(len(conc_to_lable))]
	for r, l in zip(results_by_gender['male'],labels_by_gender['male']):
		groups[l].append(r)

	confusion_matrix_male = np.zeros([len(conc_to_lable),len(conc_to_lable)])
	for l in range(len(conc_to_lable)):
		if len(groups[l]) == 0:
			continue
		predicted = groups[l]
		for p in predicted:
			confusion_matrix_male[l][p] += 1
	sum_ = confusion_matrix_male.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_male_norm = confusion_matrix_male / sum_

	groups = [[] for _ in range(len(conc_to_lable))]
	for r, l in zip(results_by_gender['female'],labels_by_gender['female']):
		groups[l].append(r)
	confusion_matrix_female = np.zeros([len(conc_to_lable),len(conc_to_lable)])
	for l in range(len(conc_to_lable)):
		if len(groups[l]) == 0:
			continue
		predicted = groups[l]
		for p in predicted:
			confusion_matrix_female[l][p] += 1
	sum_ = confusion_matrix_female.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_female_norm = confusion_matrix_female / sum_

	gender_confusion = np.abs(confusion_matrix_male_norm - confusion_matrix_female_norm)

	diag = gender_confusion.diagonal()
	total = 0
	num = 0
	# print('{}_gender_equal_odds'.format(name))
	# for i in np.argsort(diag)[::-1]:
	# 	sum_1 = np.sum(confusion_matrix_male[i])
	# 	sum_2 = np.sum(confusion_matrix_female[i])
	# 	if confusion_matrix_male[i][i] == 0 and confusion_matrix_female[i][i] == 0:
	# 		continue
	# 	print("{:25} {:.4f}({:.1f}/{:.1f}) {:.4f}({:.1f}/{:.1f})".format(conc_label_to_name[i],
	# 		confusion_matrix_male_norm[i][i],
	# 		confusion_matrix_male[i][i],
	# 		np.sum(confusion_matrix_male[i]),
	# 		confusion_matrix_female_norm[i][i],
	# 		confusion_matrix_female[i][i],
	# 		np.sum(confusion_matrix_female[i])))
	# 	total += np.abs(confusion_matrix_male_norm[i][i]-confusion_matrix_female_norm[i][i])
	# 	num += 1

	# print('='*40)
	# print("{:25} {:.4f} {:.4f}".format(name+'_difference',total,total/num))
	# print('='*40)

	fo.write('{}_gender_equal_odds\n'.format(name))
	for i in np.argsort(diag)[::-1]:
		sum_1 = np.sum(confusion_matrix_male[i])
		sum_2 = np.sum(confusion_matrix_female[i])
		if confusion_matrix_male[i][i] == 0 and confusion_matrix_female[i][i] == 0:
			continue
		fo.write("{:25} {:.4f}({:.1f}/{:.1f}) {:.4f}({:.1f}/{:.1f})\n".format(conc_label_to_name[i],
			confusion_matrix_male_norm[i][i],
			confusion_matrix_male[i][i],
			np.sum(confusion_matrix_male[i]),
			confusion_matrix_female_norm[i][i],
			confusion_matrix_female[i][i],
			np.sum(confusion_matrix_female[i])))
		total += np.abs(confusion_matrix_male_norm[i][i]-confusion_matrix_female_norm[i][i])
		num += 1
	fo.write('end_of_{}_gender_equal_odds\n'.format(name))
	fo.write('='*40+'\n')
	fo.write("{:25} {:.4f} {:.4f}\n".format(name+'_difference',total,total/num))
	fo.write('='*40+'\n')

	fo.close()

	mean_demog_parity, mean_equal_odd_avg, \
	mean_equal_odd_total, mean_demog_parity_smooth_epsilon_max, \
	mean_demog_parity_smooth_epsilon_avg = read_result(result_file,name)

	command = 'rm {}'.format(result_file)
	os.system(command)

	if loss_type == 'dp':
		return mean_demog_parity
	elif loss_type == 'eo':
		return mean_equal_odd_avg
	elif loss_type == 'both':
		return mean_demog_parity + mean_equal_odd_avg
	

def fairness_silent_detail(y_true,y_pred,users,name,loss_type):

	import time
	result_file = 'res_{}.txt'.format(time.time())

	fo = open(result_file,'w')

	y_true = np.argmax(y_true, axis=1)
	y_pred = np.argmax(y_pred, axis=1)

	conc_to_lable = json.load(open(os.path.join(FLAGS.data_prefix,'conc_to_lable.json'),'r'))
	conc_label_to_name = json.load(open(os.path.join(FLAGS.data_prefix,'conc_label_to_name.json'),'r'))

	conc_label_to_name = { int(k):conc_label_to_name[k] for k in conc_label_to_name.keys() }

	male_users = json.load(open(os.path.join(FLAGS.data_prefix,'male_users.json'),'r'))
	female_users = json.load(open(os.path.join(FLAGS.data_prefix,'female_users.json'),'r'))

	common = set(users) & set(male_users) 
	fo.write('male ' + str(len(common)) + '\n')
	
	common = set(users) & set(female_users)
	fo.write('female ' + str(len(common)) + '\n')
	
	fo.write('='*40+'\n')
	
	results_by_gender= { 'male':[], 'female':[] }
	labels_by_gender= { 'male':[], 'female':[] }
	
	for r, l, u in zip(y_pred,y_true,users):
		if u in male_users:
			results_by_gender['male'].append(r)
			labels_by_gender['male'].append(l)
		elif u in female_users:
			results_by_gender['female'].append(r)
			labels_by_gender['female'].append(l)

	male_dist = np.asarray([0.0 for _ in range(len(conc_to_lable))])
	female_dist = np.asarray([0.0 for _ in range(len(conc_to_lable))])

	# demog_disparity 

	predicted = results_by_gender['male']
	for p in predicted:
		male_dist[p] += 1

	predicted = results_by_gender['female']
	for p in predicted:
		female_dist[p] += 1


	# fo.write('zzq sum(male_dist) ' + str(sum(male_dist)) + '\n')
	# fo.write('zzq sum(female_dist) ' + str(sum(female_dist)) + '\n')

	norm_male_dist = male_dist/sum(male_dist)
	norm_female_dist = female_dist/sum(female_dist)
	
	smooth_norm_male_dist = (male_dist+1)/(sum(male_dist)+len(conc_to_lable))
	smooth_norm_female_dist = (female_dist+1)/(sum(female_dist)+len(conc_to_lable))
	
	gender_demog_parity =  0.5 * np.sum(np.abs(norm_male_dist-norm_female_dist))


	# print('{:25} {:.4f}'.format(name+'_gender_demog_parity', gender_demog_parity))
	# print("{:25} {:13} {:13}".format('concentration','male','female'))
	# for idx in np.argsort(np.abs(norm_male_dist-norm_female_dist))[::-1]:
	# 	print("{:25} {:.4f} {:.4f} {:.4f} {:.4f}".format(conc_label_to_name[idx],
	# 		norm_male_dist[idx],smooth_norm_male_dist[idx],
	# 		norm_female_dist[idx],smooth_norm_female_dist[idx]))
	# print('end_of_{}_gender_demog_parity'.format(name))
	# print('='*40)


	fo.write('{:25} {:.4f}\n'.format(name+'_gender_demog_parity', gender_demog_parity))
	fo.write("{:25} {:13} {:13}\n".format('concentration','male','female'))
	for idx in np.argsort(np.abs(norm_male_dist-norm_female_dist))[::-1]:
		fo.write("{:25} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(conc_label_to_name[idx],
			norm_male_dist[idx],smooth_norm_male_dist[idx],
			norm_female_dist[idx],smooth_norm_female_dist[idx]))
	fo.write('end_of_{}_gender_demog_parity\n'.format(name))
	fo.write('='*40+'\n')

	# equal odds
	groups = [[] for _ in range(len(conc_to_lable))]
	for r, l in zip(results_by_gender['male'],labels_by_gender['male']):
		groups[l].append(r)

	confusion_matrix_male = np.zeros([len(conc_to_lable),len(conc_to_lable)])
	for l in range(len(conc_to_lable)):
		if len(groups[l]) == 0:
			continue
		predicted = groups[l]
		for p in predicted:
			confusion_matrix_male[l][p] += 1
	sum_ = confusion_matrix_male.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_male_norm = confusion_matrix_male / sum_

	groups = [[] for _ in range(len(conc_to_lable))]
	for r, l in zip(results_by_gender['female'],labels_by_gender['female']):
		groups[l].append(r)
	confusion_matrix_female = np.zeros([len(conc_to_lable),len(conc_to_lable)])
	for l in range(len(conc_to_lable)):
		if len(groups[l]) == 0:
			continue
		predicted = groups[l]
		for p in predicted:
			confusion_matrix_female[l][p] += 1
	sum_ = confusion_matrix_female.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_female_norm = confusion_matrix_female / sum_

	gender_confusion = np.abs(confusion_matrix_male_norm - confusion_matrix_female_norm)

	diag = gender_confusion.diagonal()
	total = 0
	num = 0
	# print('{}_gender_equal_odds'.format(name))
	# for i in np.argsort(diag)[::-1]:
	# 	sum_1 = np.sum(confusion_matrix_male[i])
	# 	sum_2 = np.sum(confusion_matrix_female[i])
	# 	if confusion_matrix_male[i][i] == 0 and confusion_matrix_female[i][i] == 0:
	# 		continue
	# 	print("{:25} {:.4f}({:.1f}/{:.1f}) {:.4f}({:.1f}/{:.1f})".format(conc_label_to_name[i],
	# 		confusion_matrix_male_norm[i][i],
	# 		confusion_matrix_male[i][i],
	# 		np.sum(confusion_matrix_male[i]),
	# 		confusion_matrix_female_norm[i][i],
	# 		confusion_matrix_female[i][i],
	# 		np.sum(confusion_matrix_female[i])))
	# 	total += np.abs(confusion_matrix_male_norm[i][i]-confusion_matrix_female_norm[i][i])
	# 	num += 1

	# print('='*40)
	# print("{:25} {:.4f} {:.4f}".format(name+'_difference',total,total/num))
	# print('='*40)

	fo.write('{}_gender_equal_odds\n'.format(name))
	for i in np.argsort(diag)[::-1]:
		sum_1 = np.sum(confusion_matrix_male[i])
		sum_2 = np.sum(confusion_matrix_female[i])
		if confusion_matrix_male[i][i] == 0 and confusion_matrix_female[i][i] == 0:
			continue
		fo.write("{:25} {:.4f}({:.1f}/{:.1f}) {:.4f}({:.1f}/{:.1f})\n".format(conc_label_to_name[i],
			confusion_matrix_male_norm[i][i],
			confusion_matrix_male[i][i],
			np.sum(confusion_matrix_male[i]),
			confusion_matrix_female_norm[i][i],
			confusion_matrix_female[i][i],
			np.sum(confusion_matrix_female[i])))
		total += np.abs(confusion_matrix_male_norm[i][i]-confusion_matrix_female_norm[i][i])
		num += 1
	fo.write('end_of_{}_gender_equal_odds\n'.format(name))
	fo.write('='*40+'\n')
	fo.write("{:25} {:.4f} {:.4f}\n".format(name+'_difference',total,total/num))
	fo.write('='*40+'\n')

	fo.close()

	mean_demog_parity, mean_equal_odd_avg, \
	mean_equal_odd_total, mean_demog_parity_smooth_epsilon_max, \
	mean_demog_parity_smooth_epsilon_avg = read_result(result_file,name)

	command = 'rm {}'.format(result_file)
	os.system(command)

	return mean_demog_parity, mean_equal_odd_avg
	