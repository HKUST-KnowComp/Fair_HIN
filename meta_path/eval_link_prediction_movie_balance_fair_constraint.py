import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
import argparse
import csv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math
from collections import Counter
import json
import copy
from datetime import datetime
import time
# Global Variables
emb = {}
user_concentration = {}
all_users = {}
concentration = {}
concentration_inverse = {}
concentration_label_2_name = {}

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("Device:", device)
print("Num GPUs:", n_gpu)

class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, W):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) 
		self.activate = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, input_size)
		self.W = nn.Parameter(W,requires_grad=False)
		self.bias = nn.Parameter(torch.randn(num_classes),requires_grad=True)
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
	def forward(self, x):
		out = self.fc1(x)
		self.activate(out) # out = gelu(out) 
		out = self.fc2(out)
		self.activate(out) # out = gelu(out)
		out = torch.matmul(out, self.W) + self.bias
		return out

def load_embeddings():
	with open(os.path.join(args.emb_dir, args.emb)) as fin:
		for line in fin:
			line = line.strip().split()
			if len(line) == 2:
				vocab_size, dim = line
				args.dim = int(dim)
			else:
				emb[line[0]] = np.asarray(list(map(float,line[1:])))
				if line[0][0] == 'u':
					if line[0] not in all_users:
						all_users[line[0]] = len(all_users)

def balance_user(users,male,female):

	if len(male) == len(female):
		return users
	
	elif len(male) > len(female):
		more = len(male)
		less = len(female)
		
		ind = np.random.permutation(more)
		male = np.array(male)
		male = male[ind]
		users = list(female) + list(male[:less]) 
		
		return users

	elif len(female) > len(male):
		more = len(female)
		less = len(male)

		ind = np.random.permutation(more)
		female = np.array(female)
		female = female[ind]
		users = list(male) + list(female[:less]) 
		
		return users

def data_preprocess():
	
	train_features = []
	dev_features = []
	test_features = []

	train_labels = []
	dev_labels = []
	test_labels = []

	dev_users = []
	test_users = []

	male_users = set()
	female_users = set()

	train_conc_users = {}


	data_dir = os.path.join(args.data_dir,'split_{}'.format(args.outer_no))
	with open(os.path.join(data_dir, 'user_career_non_test.csv'), 'r') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		next(csv_reader, None)
		for row in csv_reader:
			if row == []:
				continue
			user_id, concentration_id = row
			user_id = 'u'+user_id
			concentration_id = 'c'+ concentration_id
			
			if concentration_id not in concentration:
				concentration[concentration_id] = len(concentration)
				concentration_inverse[concentration[concentration_id]] = concentration_id

	with open(os.path.join(data_dir, 'user_career_test.csv'), 'r') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		next(csv_reader, None)
		for row in csv_reader:
			if row == []:
				continue
			user_id, concentration_id = row
			user_id = 'u'+user_id
			concentration_id = 'c'+ concentration_id
			
			if concentration_id not in concentration:
				concentration[concentration_id] = len(concentration)
				concentration_inverse[concentration[concentration_id]] = concentration_id
	
	# concentration embs
	concen_embs = []
	for i in range(len(concentration)):
		concen_embs.append(emb[concentration_inverse[i]])
	concen_embs = np.asarray(concen_embs)
	args.num_classes = len(concentration) 


	with open(os.path.join(data_dir, 'user_career_test.csv'), 'r') as csvfile:
		csv_reader = csv.reader(csvfile,delimiter=',')
		next(csv_reader, None)
		for row in csv_reader:
			if row == []:
				continue
			user_id, concentration_id = row
			user_id = 'u'+user_id
			concentration_id = 'c'+ concentration_id
			if user_id in all_users:
				test_features.append(emb[user_id])
				test_labels.append(concentration[concentration_id])
				test_users.append(user_id)

	with open(os.path.join(data_dir, 'user_career_non_test_split_{}.csv'.format(args.inner_no)), 'r') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		next(csv_reader, None)
		for row in csv_reader:
			if row == []:
				continue
			user_id, concentration_id = row
			user_id = 'u'+user_id
			concentration_id = 'c'+ concentration_id
			if user_id in all_users:
				dev_features.append(emb[user_id])
				dev_labels.append(concentration[concentration_id])
				dev_users.append(user_id)

	for i in range(args.num_folds_inner):
		if i == args.inner_no:
			continue
		with open(os.path.join(data_dir, 'user_career_non_test_split_{}.csv'.format(i)), 'r') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			next(csv_reader, None)
			for row in csv_reader:
				if row == []:
					continue
				user_id, concentration_id = row
				user_id = 'u'+user_id
				concentration_id = 'c'+ concentration_id
				if user_id in all_users:
					if concentration_id not in train_conc_users:
						train_conc_users[concentration_id] = []
					train_conc_users[concentration_id].append(user_id)

	for concentration_id in train_conc_users:
		users = train_conc_users[concentration_id]
		male = list(male_users & set(users))
		female = list(female_users & set(users))
		users = balance_user(users, male, female)
		for user_id in users:
			train_features.append(emb[user_id])
			train_labels.append(concentration[concentration_id])


	with open(os.path.join(args.data_dir,'career_id_to_name.csv'),'r') as csvfile:
		csv_reader = csv.reader(csvfile,delimiter=',')
		next(csv_reader, None)
		for row in csv_reader:
			if row == []:
				continue
			concentration_id, concentration_name = row
			concentration_id = 'c'+concentration_id
			if concentration_id in concentration:
				concentration_label_2_name[concentration[concentration_id]] = concentration_name
	print(Counter(train_labels))
	# from imblearn.over_sampling import SMOTE
	# sm = SMOTE(random_state=42)
	# train_features_res, train_labels_res = sm.fit_resample(train_features, train_labels)
	return train_features, train_labels, dev_features, dev_labels, dev_users, test_features, test_labels, test_users, concen_embs

def fairness(result,labels,users,name):
	male_user = set()
	female_user = set()
	with open(os.path.join(args.data_dir,'demog.csv'), mode='r') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			next(csv_reader, None)
			for row in csv_reader:
				if row == []:
					continue
				else:
					user_id, gender, age = row
					user_id = "u"+user_id
					if gender == '0':
						male_user.add(user_id)
					elif gender == '1':
						female_user.add(user_id)
	male_user = list(male_user)
	female_users = list(female_user)
	
	print("male {} female {}".format(len(male_user),
		len(female_user) ))
	
	print('='*40)
	results_by_gender= { 'male':[], 'female':[] }
	labels_by_gender= { 'male':[], 'female':[] }

	for r, l, u in zip(result,labels,users):
		if u in male_user:
			results_by_gender['male'].append(r)
			labels_by_gender['male'].append(l)
		elif u in female_user:
			results_by_gender['female'].append(r)
			labels_by_gender['female'].append(l)

	male_dist = np.asarray([0.0 for _ in range(len(concentration))])
	female_dist = np.asarray([0.0 for _ in range(len(concentration))])

	# demog_disparity 

	predicted = np.argmax(results_by_gender['male'], 1)
	for p in predicted:
		male_dist[p] += 1

	predicted = np.argmax(results_by_gender['female'], 1)
	for p in predicted:
		female_dist[p] += 1


	norm_male_dist = male_dist/sum(male_dist)
	norm_female_dist = female_dist/sum(female_dist)
	
	smooth_norm_male_dist = (male_dist+1)/(sum(male_dist)+len(concentration))
	smooth_norm_female_dist = (female_dist+1)/(sum(female_dist)+len(concentration))
	

	gender_demog_parity =  0.5 * np.sum(np.abs(norm_male_dist-norm_female_dist))

	print('{}_gender_demog_parity {:.4f}'.format(name, gender_demog_parity))
	print("{:25} {:13} {:13}".format('concentration','male','female'))
	for idx in np.argsort(np.abs(norm_male_dist-norm_female_dist))[::-1]:
		print("{:25} {:.4f} {:.4f} {:.4f} {:.4f}".format(concentration_label_2_name[idx],
			norm_male_dist[idx],smooth_norm_male_dist[idx],
			norm_female_dist[idx],smooth_norm_female_dist[idx]))

	print('end_of_{}_gender_demog_parity'.format(name))
	print('='*40)

	# equal odds
	groups = [[] for _ in range(len(concentration))]
	for r, l in zip(results_by_gender['male'],labels_by_gender['male']):
		groups[l].append(r)

	confusion_matrix_male = np.zeros([len(concentration),len(concentration)])
	for l in range(len(concentration)):
		if len(groups[l]) == 0:
			continue
		predicted = np.argmax(groups[l],1)
		for p in predicted:
			confusion_matrix_male[l][p] += 1
	sum_ = confusion_matrix_male.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_male_norm = confusion_matrix_male / sum_

	groups = [[] for _ in range(len(concentration))]
	for r, l in zip(results_by_gender['female'],labels_by_gender['female']):
		groups[l].append(r)
	confusion_matrix_female = np.zeros([len(concentration),len(concentration)])
	for l in range(len(concentration)):
		if len(groups[l]) == 0:
			continue
		predicted = np.argmax(groups[l],1)
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
		print("{:25} {:.4f}({:.1f}/{:.1f}) {:.4f}({:.1f}/{:.1f})".format(concentration_label_2_name[i],
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


def fairness_salient(result,labels,users):
	male_user = set()
	female_user = set()
	with open(os.path.join(args.data_dir,'demog.csv'), mode='r') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			next(csv_reader, None)
			for row in csv_reader:
				if row == []:
					continue
				else:
					user_id, gender, age = row
					user_id = "u"+user_id
					if gender == '0':
						male_user.add(user_id)
					elif gender == '1':
						female_user.add(user_id)
	male_user = list(male_user)
	female_users = list(female_user)

	results_by_gender= { 'male':[], 'female':[] }
	labels_by_gender= { 'male':[], 'female':[] }

	for r, l, u in zip(result,labels,users):
		if u in male_user:
			results_by_gender['male'].append(r)
			labels_by_gender['male'].append(l)
		elif u in female_user:
			results_by_gender['female'].append(r)
			labels_by_gender['female'].append(l)

	male_dist = np.asarray([0.0 for _ in range(len(concentration))])
	female_dist = np.asarray([0.0 for _ in range(len(concentration))])

	# demog_disparity 

	predicted = np.argmax(results_by_gender['male'], 1)
	for p in predicted:
		male_dist[p] += 1

	predicted = np.argmax(results_by_gender['female'], 1)
	for p in predicted:
		female_dist[p] += 1


	norm_male_dist = male_dist/sum(male_dist)
	norm_female_dist = female_dist/sum(female_dist)
	
	smooth_norm_male_dist = (male_dist+1)/(sum(male_dist)+len(concentration))
	smooth_norm_female_dist = (female_dist+1)/(sum(female_dist)+len(concentration))
	
	gender_demog_parity =  0.5 * np.sum(np.abs(norm_male_dist-norm_female_dist))

	# equal odds
	groups = [[] for _ in range(len(concentration))]
	for r, l in zip(results_by_gender['male'],labels_by_gender['male']):
		groups[l].append(r)

	confusion_matrix_male = np.zeros([len(concentration),len(concentration)])
	for l in range(len(concentration)):
		if len(groups[l]) == 0:
			continue
		predicted = np.argmax(groups[l],1)
		for p in predicted:
			confusion_matrix_male[l][p] += 1
	sum_ = confusion_matrix_male.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_male_norm = confusion_matrix_male / sum_

	groups = [[] for _ in range(len(concentration))]
	for r, l in zip(results_by_gender['female'],labels_by_gender['female']):
		groups[l].append(r)
	confusion_matrix_female = np.zeros([len(concentration),len(concentration)])
	for l in range(len(concentration)):
		if len(groups[l]) == 0:
			continue
		predicted = np.argmax(groups[l],1)
		for p in predicted:
			confusion_matrix_female[l][p] += 1
	sum_ = confusion_matrix_female.sum(axis=1,keepdims=True)
	sum_[sum_==0] = 1e-5
	confusion_matrix_female_norm = confusion_matrix_female / sum_

	gender_confusion = np.abs(confusion_matrix_male_norm - confusion_matrix_female_norm)

	diag = gender_confusion.diagonal()
	total = 0
	num = 0
	for i in np.argsort(diag)[::-1]:
		sum_1 = np.sum(confusion_matrix_male[i])
		sum_2 = np.sum(confusion_matrix_female[i])
		if confusion_matrix_male[i][i] == 0 and confusion_matrix_female[i][i] == 0:
			continue
		total += np.abs(confusion_matrix_male_norm[i][i]-confusion_matrix_female_norm[i][i])
		num += 1
	equalized_odd = total/num
	return gender_demog_parity, equalized_odd


def mean_reciprocal_rank(logits, labels):
	mrr = 0.0
	for i in range(len(labels)):
		index = np.argsort(logits[i])[::-1]
		mrr += 1/ (index.tolist().index(labels[i])+1)
	return mrr / len(labels)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--emb',type=str, default='w200.l200.cupuc.upu.w200.l200.txt')

	parser.add_argument('--batch_size',type=int, default=32)

	parser.add_argument('--dim',type=int, default=128)

	parser.add_argument('--epoch',type=int, default=100)

	parser.add_argument('--lr',type=float, default=0.0001)

	parser.add_argument('--l2',type=float, default=0.01)

	parser.add_argument('--hidden_size',type=int, default=64)

	parser.add_argument('--data_dir',type=str, default="MovieLens")

	parser.add_argument('--emb_dir',type=str, default="emb") 

	parser.add_argument('--save_dir',type=str, default="save_model") 

	parser.add_argument('--method',type=str, default="m2v_balance")

	parser.add_argument('--inner_no',type=int, default=0, choices=range(4))

	parser.add_argument('--outer_no',type=int, default=0, choices=range(3))

	parser.add_argument('--thres_file',type=str, default='thres_fair_conditions.json')

	parser.add_argument('--num_folds_inner',type=int, default=4)

	args = parser.parse_args()

	# threshold = {}
	# threshold['dp'] = {'dev_high':0.1084,'dev_med':0.1626,'dev_low':0.2168,\
	# 				'test_high':0.0992,'test_med':0.1488,'test_low':0.1984}
	# threshold['eo'] = {'dev_high':0.0590,'dev_med':0.0884,'dev_low':0.1179,\
	# 				'test_high':0.0524,'test_med':0.0786,'test_low':0.1047}

	with open(args.thres_file, 'r') as fin:
		threshold = json.load(fin)

	load_embeddings()			

	train_features, train_labels, dev_features, dev_labels, dev_users, test_features, test_labels, test_users, concen_embs = data_preprocess()

	net = Net(args.dim, args.hidden_size, args.num_classes, torch.tensor(concen_embs,dtype=torch.float32).view(args.dim,-1).to(device))
	net.to(device)   

	train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_features,dtype=torch.float), torch.tensor(train_labels,dtype=torch.long))
	dev_dataset = torch.utils.data.TensorDataset(torch.tensor(dev_features,dtype=torch.float), torch.tensor(dev_labels,dtype=torch.long))
	test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_features,dtype=torch.float), torch.tensor(test_labels,dtype=torch.long))

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
								batch_size=args.batch_size, 
								shuffle=True)

	dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, 
								batch_size=args.batch_size, 
								shuffle=False)

	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
								batch_size=args.batch_size, 
								shuffle=False)
	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()  
	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)  

	# for name, param in net.named_parameters():
	# 	if param.requires_grad:
	# 		print (name, param.size())

	# Train the Model
	time_stamp = time.time()
	mrr_best = {'dp_high':0,'dp_med':0,'dp_low':0,'eo_high':0,'eo_med':0,'eo_low':0}
	
	for epoch in range(args.epoch):
		loss_per_epoch = 0.0
		sample_per_epoch = 0.0
		for i, (users, labels) in enumerate(train_loader):  
			# Forward + Backward + Optimize
			optimizer.zero_grad()  # zero the gradient buffer
			outputs = net(users.to(device))
			loss = criterion(outputs, labels.to(device))
			loss_per_epoch += loss.item()
			sample_per_epoch += 1
			loss.backward()
			optimizer.step()

		dev_result = []
		dev_predict = []
		for users, labels in dev_loader:
			outputs = net(users.to(device))
			dev_result.extend(outputs.detach().cpu().numpy().tolist())
			_, predicted = torch.max(outputs.detach(), 1) # outputs.data
			dev_predict.extend(predicted.cpu().numpy().tolist())

		test_result = []
		test_predict = []
		for users, labels in test_loader:
			outputs = net(users.to(device))
			test_result.extend(outputs.detach().cpu().numpy().tolist())
			_, predicted = torch.max(outputs.detach(), 1) # outputs.data
			test_predict.extend(predicted.cpu().numpy().tolist())

		# train_result = []
		# train_predict = []
		# train_label_temp = []
		# for users, labels in train_loader:
		# 	outputs = net(users.to(device))
		# 	train_result.extend(outputs.detach().cpu().numpy().tolist())
		# 	_, predicted = torch.max(outputs.detach(), 1) # outputs.data
		# 	train_predict.extend(predicted.cpu().numpy().tolist())
		# 	train_label_temp.extend(labels.cpu().numpy().tolist())
		
		mrr_dev = mean_reciprocal_rank(dev_result,dev_labels)
		dp_dev, eo_dev = fairness_salient(dev_result,dev_labels,dev_users)

		mrr_test = mean_reciprocal_rank(test_result,test_labels)
		dp_test, eo_test = fairness_salient(test_result,test_labels,test_users)


		print ('Epoch: [{:2d}/{}], Loss: {:.4f}, Dev Acc: {:.4f}, Dev F1: {:.4f}, Dev MRR: {:.4f}' .format(
			epoch+1, args.epoch, loss_per_epoch/sample_per_epoch, 
			# accuracy_score(train_label_temp,train_predict),  Train Acc {:.4f}
			accuracy_score(dev_labels,dev_predict),
			f1_score(dev_labels, dev_predict, average='weighted'), mrr_dev))
		
		for fair_level in ['low','med','high']:
			if (dp_dev < threshold['dp']['dev_'+fair_level] \
			and dp_test < threshold['dp']['test_'+fair_level] \
			and mrr_dev > mrr_best['dp_'+fair_level]) or epoch == 0:
				mrr_best['dp_'+fair_level] = mrr_dev
				
				if not os.path.exists(args.save_dir):
					os.makedirs(args.save_dir)
				torch.save(net.state_dict(), os.path.join(args.save_dir,'ml_model_dp_{}_{}.pkl'.format(fair_level,time_stamp)))
				

			if (eo_dev < threshold['eo']['dev_'+fair_level] \
			and eo_test < threshold['eo']['test_'+fair_level] \
			and mrr_dev > mrr_best['eo_'+fair_level]) or epoch == 0:
				mrr_best['eo_'+fair_level] = mrr_dev
				
				if not os.path.exists(args.save_dir):
					os.makedirs(args.save_dir)
				torch.save(net.state_dict(), os.path.join(args.save_dir,'ml_model_eo_{}_{}.pkl'.format(fair_level,time_stamp)))
				

	for fair_level in ['low','med','high']:
		net.load_state_dict(torch.load(os.path.join(args.save_dir,'ml_model_dp_{}_{}.pkl'.format(fair_level,time_stamp))))
		dev_result = []
		dev_predict = []
		for users, labels in dev_loader:
			outputs = net(users.to(device))
			dev_result.extend(outputs.detach().cpu().numpy().tolist())
			_, predicted = torch.max(outputs.detach(), 1) # outputs.data
			dev_predict.extend(predicted.cpu().numpy().tolist())

		test_result = []
		test_predict = []
		for users, labels in test_loader:
			outputs = net(users.to(device))
			test_result.extend(outputs.detach().cpu().numpy().tolist())
			_, predicted = torch.max(outputs.detach(), 1) # outputs.data
			test_predict.extend(predicted.cpu().numpy().tolist())

		mrr_dev = mean_reciprocal_rank(dev_result,dev_labels)
		dp_dev, eo_dev = fairness_salient(dev_result,dev_labels,dev_users)

		mrr_test = mean_reciprocal_rank(test_result,test_labels)
		dp_test, eo_test = fairness_salient(test_result,test_labels,test_users)

		
		with open('{}_dp_{}_ml.txt'.format(args.method,fair_level),'a') as fo:
			fo.write('emb {} outer_no {:2d} inner_no {:2d}\n'.format(
				args.emb, args.outer_no, args.inner_no))
			fo.write('dev_mrr {:.5f} dev_dp {:.5f}\n'.format(mrr_dev, dp_dev))
			fo.write('test_mrr {:.5f} test_dp {:.5f}\n'.format(mrr_test, dp_test))

		######################################

		net.load_state_dict(torch.load(os.path.join(args.save_dir,'ml_model_eo_{}_{}.pkl'.format(fair_level,time_stamp))))
		dev_result = []
		dev_predict = []
		for users, labels in dev_loader:
			outputs = net(users.to(device))
			dev_result.extend(outputs.detach().cpu().numpy().tolist())
			_, predicted = torch.max(outputs.detach(), 1) # outputs.data
			dev_predict.extend(predicted.cpu().numpy().tolist())

		test_result = []
		test_predict = []
		for users, labels in test_loader:
			outputs = net(users.to(device))
			test_result.extend(outputs.detach().cpu().numpy().tolist())
			_, predicted = torch.max(outputs.detach(), 1) # outputs.data
			test_predict.extend(predicted.cpu().numpy().tolist())

		mrr_dev = mean_reciprocal_rank(dev_result,dev_labels)
		dp_dev, eo_dev = fairness_salient(dev_result,dev_labels,dev_users)

		mrr_test = mean_reciprocal_rank(test_result,test_labels)
		dp_test, eo_test = fairness_salient(test_result,test_labels,test_users)

		
		with open('{}_eo_{}_ml.txt'.format(args.method,fair_level),'a') as fo:
			fo.write('emb {} outer_no {:2d} inner_no {:2d}\n'.format(
				args.emb, args.outer_no, args.inner_no))
			fo.write('dev_mrr {:.5f} dev_eo {:.5f}\n'.format(mrr_dev, eo_dev))
			fo.write('test_mrr {:.5f} test_eo {:.5f}\n'.format(mrr_test, eo_test))



	for fair_level in ['low','med','high']:
		command = 'rm {}'.format(os.path.join(args.save_dir,'ml_model_dp_{}_{}.pkl'.format(fair_level,time_stamp)))
		print(command)
		os.system(command)
		command = 'rm {}'.format(os.path.join(args.save_dir,'ml_model_eo_{}_{}.pkl'.format(fair_level,time_stamp)))
		print(command)
		os.system(command)
