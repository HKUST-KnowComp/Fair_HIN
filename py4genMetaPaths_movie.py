# large dev test
import sys
import os
import random
from collections import Counter
import csv
import argparse
import math
import numpy as np
import re
import random

class MetaPathGenerator:
	def __init__(self):
		self.id_user = dict()
		self.id_page = dict()
		self.id_concentration = dict()
		self.user_page = dict()
		self.page_user = dict()
		self.user_concentration = dict()
		self.concentration_user = dict()

		self.male_users = set()
		self.female_users = set()

		self.cupuc_users = set()

		self.page_dist = {}
	

	def read_data(self):

		with open(os.path.join(args.data_dir, 'user_career_emb.csv'), 'r') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			next(csv_reader, None)
			for row in csv_reader:
				if row == []:
					continue
				user_id, concentration_id = row
				user_id = 'u'+user_id
				concentration_id = 'c'+ concentration_id
				if user_id not in self.user_concentration:
					self.user_concentration[user_id] = []
				self.user_concentration[user_id].append(concentration_id)
				
				if concentration_id not in self.concentration_user:
					self.concentration_user[concentration_id] = []
				self.concentration_user[concentration_id].append(user_id)


		with open(os.path.join(args.data_dir, 'user_moive.csv'), 'r') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			next(csv_reader, None)
			for row in csv_reader:
				if row == []:
					continue
				user_id, page_id = row
				user_id = 'u'+user_id
				page_id = 'p'+ page_id

				if user_id not in self.user_page:
					self.user_page[user_id] = []
				self.user_page[user_id].append(page_id)
	
				if page_id not in self.page_user:
					self.page_user[page_id] = []
				self.page_user[page_id].append(user_id)

		
		num_pages_per_user = []
		for user in self.user_page:
			num_pages_per_user.append(len(self.user_page[user]))
		

		num_users_per_page = []
		for page in self.page_user:
			num_users_per_page.append(len(self.page_user[page]))


		num_users_per_conc = []
		for conc in self.concentration_user:
			num_users_per_conc.append(len(self.concentration_user[conc]))

		print("{} users".format(len(self.user_page)))
		print("{} movies".format(len(self.page_user)))
		print("{} concentrations".format(len(self.concentration_user)))
		print("{:.2f} num_pages_per_user".format(sum(num_pages_per_user)/len(self.user_page)))
		print("{:.2f} num_users_per_page".format(sum(num_users_per_page)/len(self.page_user)))
		print("{:.2f} num_users_per_conc".format(sum(num_users_per_conc)/len(self.concentration_user)))

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
						self.male_users.update([user_id])
					elif gender == '1':
						self.female_users.update([user_id])
		all_users = set(self.user_page.keys())
		print("male {} female {}".format(len(self.male_users & all_users),len(self.female_users & all_users)))
		# if not os.path.exists(args.data_dir,'male_user.json'):
		# 	json.dump(list(self.male.json),open(os.path.join(args.data_dir,'male.json'),'w'))
		
		# if not os.path.exists(args.data_dir,'female_user.json'):
		# 	json.dump(list(self.female.json),open(os.path.join(args.data_dir,'female.json'),'w'))


	def read_balance_data(self):
		np.random.seed(1234)
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
						self.male_users.update([user_id])
					elif gender == '1':
						self.female_users.update([user_id])

		with open(os.path.join(args.data_dir, 'user_career_emb.csv'), 'r') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			next(csv_reader, None)
			for row in csv_reader:
				if row == []:
					continue
				user_id, concentration_id = row
				user_id = 'u'+user_id
				concentration_id = 'c'+ concentration_id
				if user_id not in self.user_concentration:
					self.user_concentration[user_id] = []
				self.user_concentration[user_id].append(concentration_id)
				
				if concentration_id not in self.concentration_user:
					self.concentration_user[concentration_id] = []
				self.concentration_user[concentration_id].append(user_id)

		b_concentration_user = dict()
		b_user_concentration = dict()
		for concentration_id in self.concentration_user:
			users = self.concentration_user[concentration_id]
			male = list(self.male_users & set(users))
			female = list(self.female_users & set(users))
			users = self.balance_user(users,male,female)
			b_concentration_user[concentration_id] = users
			for u in users:
				b_user_concentration[u] = [concentration_id]

		self.concentration_user = b_concentration_user
		self.user_concentration = b_user_concentration

		with open(os.path.join(args.data_dir, 'user_moive.csv'), 'r') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			next(csv_reader, None)
			for row in csv_reader:
				if row == []:
					continue
				user_id, page_id = row
				user_id = 'u'+user_id
				page_id = 'p'+ page_id

				if user_id not in self.user_page:
					self.user_page[user_id] = []
				self.user_page[user_id].append(page_id)
	
				if page_id not in self.page_user:
					self.page_user[page_id] = []
				self.page_user[page_id].append(user_id)

		
		num_pages_per_user = []
		for user in self.user_page:
			num_pages_per_user.append(len(self.user_page[user]))
		

		num_users_per_page = []
		for page in self.page_user:
			num_users_per_page.append(len(self.page_user[page]))


		num_users_per_conc = []
		for conc in self.concentration_user:
			num_users_per_conc.append(len(self.concentration_user[conc]))

		print("{} users".format(len(self.user_page)))
		print("{} movies".format(len(self.page_user)))
		print("{} concentrations".format(len(self.concentration_user)))
		print("{:.2f} num_pages_per_user".format(sum(num_pages_per_user)/len(self.user_page)))
		print("{:.2f} num_users_per_page".format(sum(num_users_per_page)/len(self.page_user)))
		print("{:.2f} num_users_per_conc".format(sum(num_users_per_conc)/len(self.concentration_user)))

		all_users = set(self.user_page.keys())
		print("male {} female {}".format(len(self.male_users & all_users),len(self.female_users & all_users)))
		# if not os.path.exists(args.data_dir,'male_user.json'):
		# 	json.dump(list(self.male.json),open(os.path.join(args.data_dir,'male.json'),'w'))
		
		# if not os.path.exists(args.data_dir,'female_user.json'):
		# 	json.dump(list(self.female.json),open(os.path.join(args.data_dir,'female.json'),'w'))


	def get_user_with_bias(self, users):
		num_users = len(users)

		male_users = list(set(users)&self.male_users)
		female_users = list(set(users)&self.female_users)
		unknow_users = list(set(users) - set(male_users) - set(female_users))
		
		num_male = len(male_users)
		num_female = len(female_users)
		
		num_unknown = num_users - num_male - num_female
		
		assert len(unknow_users) == num_unknown

		if num_male == 0:
			p_male = 0
			p_female = 1
		elif num_female == 0:
			p_male = 1
			p_female = 0
		else:
			if num_male > num_female:
				p_male = 1 / num_male
				p_female = args.ratio / num_female
			
			elif num_male < num_female:
				p_male = args.ratio / num_male
				p_female = 1 / num_female

			else:
				p_male = 1 / num_male
				p_female = 1 / num_female

		if num_unknown == 0:
			p_unknown = 0
		else:
			if num_male > num_female:
				p_unknown = p_male
			else:
				p_unknown = p_female
		
		sum_ = (p_male + p_female + p_unknown)
		p_male = p_male / sum_
		p_female = p_female / sum_
		p_unknown = p_unknown / sum_
		
		p = random.random()

		if num_male == 0 and num_female == 0:
			random_user_idx = random.randrange(num_unknown)
			user = unknow_users[random_user_idx]
		
		elif num_male == 0:
			if p < p_unknown:
				random_user_idx = random.randrange(num_unknown)
				user = unknow_users[random_user_idx]
			elif p < p_unknown+p_female:
				random_user_idx = random.randrange(num_female)
				user = female_users[random_user_idx]

		elif num_female == 0:
			if p < p_unknown:
				random_user_idx = random.randrange(num_unknown)
				user = unknow_users[random_user_idx]
			elif p < p_unknown+p_male:
				random_user_idx = random.randrange(num_male)
				user = male_users[random_user_idx]
		else:
			if p < p_unknown:
				random_user_idx = random.randrange(num_unknown)
				user = unknow_users[random_user_idx]
			elif p < p_unknown+p_female:
				random_user_idx = random.randrange(num_female)
				user = female_users[random_user_idx]
			else:
				random_user_idx = random.randrange(num_male)
				user = male_users[random_user_idx]

		return user
	

	def get_page_with_bias_by_user(self, user):
		pages, dist = self.page_dist[user]
		p = random.random()
		random_page_idx = [i for i, d in enumerate(dist) if p <= d][0]
		page = pages[random_page_idx]

		return page


	def generate_random_cupuc(self):
		with open(os.path.join(args.output_data_dir, args.output), 'w') as outfile:
			for start_conc in self.concentration_user:
				for j in range(0, args.walks): #wnum walks
					outline = start_conc
					cenc = start_conc
					for i in range(0, args.length):
						users = self.concentration_user[cenc]
						num_users = len(users)
						random_user_idx = random.randrange(num_users)
						user = users[random_user_idx]
						outline += " " + user

						self.cupuc_users.update([user])

						pages = self.user_page[user]
						num_pages = len(pages)
						random_page_idx = random.randrange(num_pages)
						page = pages[random_page_idx]
						outline += " " + page

						users = self.page_user[page]
						users = list(set(users) & set(self.user_concentration.keys()))
						num_users = len(users)
						if num_users == 0:
							break
						else:
							random_user_idx = random.randrange(num_users)
							user = users[random_user_idx]	
							outline += " " + user

						self.cupuc_users.update([user])

						cencs = self.user_concentration[user]
						num_cencs = len(cencs)
						random_cenc_idx = random.randrange(num_cencs)
						cenc = cencs[random_cenc_idx]
						outline += " " + cenc

					outfile.write(outline + "\n")


	def generate_random_conc_biased_cupuc(self):
		with open(os.path.join(args.output_data_dir, args.output), 'w') as outfile:
			for start_conc in self.concentration_user:
				for j in range(0, args.walks): #wnum walks
					outline = start_conc
					cenc = start_conc
					for i in range(0, args.length):
						users = self.concentration_user[cenc]
						user = self.get_user_with_bias(users)
						outline += " " + user

						self.cupuc_users.update([user])

						pages = self.user_page[user]
						num_pages = len(pages)
						random_page_idx = random.randrange(num_pages)
						page = pages[random_page_idx]
						outline += " " + page

						users = self.page_user[page]
						users = list(set(users) & set(self.user_concentration.keys()))
						if len(users) == 0:
							break
						else:
							user = self.get_user_with_bias(users)
							outline += " " + user

						self.cupuc_users.update([user])

						cencs = self.user_concentration[user]
						num_cencs = len(cencs)
						random_cenc_idx = random.randrange(num_cencs)
						cenc = cencs[random_cenc_idx]
						outline += " " + cenc

					outfile.write(outline + "\n")


	def generate_random_upu(self):
		upu_users = set(self.user_page.keys()) - self.cupuc_users

		with open(os.path.join(args.output_data_dir, args.output), 'a') as outfile:
			for start_user in upu_users:
				for j in range(0, args.upu_walks):
					outline = start_user
					user = start_user
					for i in range(0, args.upu_length):
						pages = self.user_page[user]
						num_pages = len(pages)
						random_page_idx = random.randrange(num_pages)
						page = pages[random_page_idx]
						outline += " " + page

						users = self.page_user[page]
						num_users = len(users)
						random_user_idx = random.randrange(num_users)
						user = users[random_user_idx]
						outline += " " + user

					outfile.write(outline + "\n")

					
	def balance_user(self,users,male,female):

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


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--walks', type=int, default=50)

	parser.add_argument('--length',type=int, default=50)

	parser.add_argument('--upu_walks',type=int, default=50)

	parser.add_argument('--upu_length',type=int, default=50)

	parser.add_argument('--ratio',type=int, default=1)

	parser.add_argument('--c',action='store_true')

	parser.add_argument('--b_c',action='store_true')

	parser.add_argument('--balance',action='store_true')

	parser.add_argument('--data_dir',type=str, default="MovieLens")

	parser.add_argument('--output_data_dir',type=str, default=".")

	parser.add_argument('--emb_dir',type=str, default=".")


	args = parser.parse_args()

	if not os.path.exists(args.output_data_dir):
		os.makedirs(args.output_data_dir)

	if not os.path.exists(args.emb_dir):
		os.makedirs(args.emb_dir)

	elif args.c and not args.balance:
		args.output = "w{}.l{}.cupuc.upu.w{}.l{}.txt".format(args.walks, args.length, args.upu_walks, args.upu_length)
	elif args.c and args.balance:
		args.output = "w{}.l{}.cupuc.upu.w{}.l{}.balance.txt".format(args.walks, args.length, args.upu_walks, args.upu_length)
	elif args.b_c:
		args.output = "w{}.l{}.b.cupuc.upu.w{}.l{}.r{}.txt".format(args.walks, args.length, args.upu_walks, args.upu_length, args.ratio)
	
	mpg = MetaPathGenerator()
	if args.balance:
		print('read balance data')
		mpg.read_balance_data()
	else:
		mpg.read_data()
	
	if args.c:
		mpg.generate_random_cupuc()
		mpg.generate_random_upu()
	elif args.b_c:
		mpg.generate_random_conc_biased_cupuc()
		mpg.generate_random_upu()


	command = "./metapath2vec -output {} \
-train {} -pp 1 -size 128 -window 5 -negative 5 -threads 30 -iter 5".format(
	os.path.join(args.emb_dir,args.output[:-4]), 
	os.path.join(args.output_data_dir,args.output))
	print(command)
	os.system(command)


	command = "\nrm {}".format(os.path.join(args.output_data_dir,args.output))
	print(command)
	os.system(command)





























