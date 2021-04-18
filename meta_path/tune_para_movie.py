import os
import numpy as np
import argparse
import math
from os.path import join, exists

def train_bias(num_walks, walk_len, ratio):

    name = "w{}.l{}.r{}".format(num_walks,walk_len,ratio)
    result_file = 'res_w{}.l{}.b.cupuc.upu.w{}.l{}.r{}.txt'.format(num_walks,walk_len,num_walks,walk_len,ratio)
    result_file = os.path.join(args.result_dir, result_file)
    emb_file = "w{}.l{}.b.cupuc.upu.w{}.l{}.r{}.txt".format(num_walks,walk_len,num_walks,walk_len,ratio)

    if not os.path.exists(os.path.join(args.emb_dir,emb_file)):
        command = "python py4genMetaPaths_movie.py \
--walks {} --length {} --upu_walks {} --upu_length {} --ratio {} \
--output_data_dir {} --emb_dir {} \
--data_dir {} --b_c".format(
                num_walks, walk_len, num_walks ,walk_len, ratio, \
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
                command = "CUDA_VISIBLE_DEVICES={} python eval_link_prediction_movie_fair_constraint.py --emb {} \
        --emb_dir {} --data_dir {} --num_folds_inner {} --inner_no {} --outer_no {} --method m2v_bias >> {}".format(
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

def train_default(num_walks, walk_len):
    
    name = "w{}.l{}".format(num_walks,walk_len)
    result_file = 'res_w{}.l{}.cupuc.upu.w{}.l{}.txt'.format(num_walks,walk_len,num_walks,walk_len)
    result_file = os.path.join(args.result_dir, result_file)
    emb_file = "w{}.l{}.cupuc.upu.w{}.l{}.txt".format(num_walks,walk_len,num_walks,walk_len)

    if not os.path.exists(os.path.join(args.emb_dir,emb_file)): 
        command = "python py4genMetaPaths_movie.py \
--walks {} --length {} --upu_walks {} --upu_length {} \
--output_data_dir {} --emb_dir {} \
--data_dir {} --c".format(
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
                command = "CUDA_VISIBLE_DEVICES={} python eval_link_prediction_movie_fair_constraint.py --emb {} \
        --emb_dir {} --data_dir {} --num_folds_inner {} --inner_no {} --outer_no {} --method m2v_default >> {}".format(
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
    parser.add_argument('--output_data_dir',type=str, default="/data/zzengae/code_metapath2vec/path/movie")
    parser.add_argument('--emb_dir',type=str, default="/data/zzengae/code_metapath2vec/emb/movie")
    parser.add_argument('--data_dir',type=str, default='MovieLens')
    parser.add_argument('--result_dir',type=str, default='')
    parser.add_argument('--gpu',type=int, default=0)
    parser.add_argument('--method',type=str, default='default')
    parser.add_argument('--num_folds_outer',type=int, default=3)
    parser.add_argument('--num_folds_inner',type=int, default=4)
    
    args = parser.parse_args()

    if args.method == 'default':
        args.result_dir = 'result_movie_default_cv'
        if not exists(args.result_dir):
            os.makedirs(args.result_dir)
        for num_walks in [50,100,150,200]:
            for walk_len in [50,100,150,200]:
                train_default(num_walks,walk_len)

    elif args.method == 'bias':
        args.result_dir = 'result_movie_bias_cv'
        if not exists(args.result_dir):
            os.makedirs(args.result_dir)
        for ratio in [1,2,3,4,5,6,7,8,9,10]: # 1,2,3,4,5,6,7,8,9,10
            for num_walks in [50,100,150,200]:
                for walk_len in [50,100,150,200]:
                    train_bias(num_walks,walk_len,ratio)
            


