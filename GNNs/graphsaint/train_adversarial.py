from graphsaint.globals import *
from graphsaint.inits import *
from graphsaint.model_adv import GraphSAINT
from graphsaint.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from tensorflow.python.client import timeline

import sys, os, random
import tensorflow as tf
import numpy as np
import time
import pdb
import json
from os.path import join, exists

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def evaluate_full_batch(sess,model,minibatch_iter,many_runs_timeline,mode,final=False):
    """
    Full batch evaluation
    NOTE: HERE GCN RUNS THROUGH THE FULL GRAPH. HOWEVER, WE CALCULATE F1 SCORE
        FOR VALIDATION / TEST NODES ONLY. 
    """
    #return 0,0,0,0
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    t1 = time.time()
    num_cls = minibatch_iter.class_arr.shape[-1]
    feed_dict, labels = minibatch_iter.feed_dict(mode)
    if FLAGS.timeline:
        preds, loss, disc_acc = sess.run([model.encoder_preds, model.encoder_loss, model.disc_acc], feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.append(chrome_trace)
    else:
        preds, loss, disc_acc= sess.run([model.encoder_preds, model.encoder_loss, model.disc_acc], feed_dict=feed_dict)
    node_val_test = minibatch_iter.node_val if mode=='val' else minibatch_iter.node_test
    t2 = time.time()
    
    mrr = mean_reciprocal_rank(labels[node_val_test],preds[node_val_test])

    f1_weight = calc_f1(labels[node_val_test],preds[node_val_test])
    if mode == 'val':
        name = 'dev'
    elif mode == 'test':
        name = 'test'

    dp = fairness_silent(labels[node_val_test],preds[node_val_test],node_val_test,name,'dp')
    eo = fairness_silent(labels[node_val_test],preds[node_val_test],node_val_test,name,'eo')
    
    if final:
        fairness(labels[node_val_test],preds[node_val_test],node_val_test,name)
        print('Tag_{}, Mrr {:.5f}'.format(name.title(),mrr))
    
    return loss, mrr, f1_weight, disc_acc, dp, eo


def construct_placeholders(num_classes):
    placeholders = {
        'alpha': tf.placeholder(tf.float32, shape=(None), name='alpha'),
        'gamma': tf.placeholder(tf.float32, shape=(None), name='gamma'),
        'lr': tf.placeholder(tf.float32, shape=(), name='lr'),
        'labels': tf.placeholder(DTYPE, shape=(None, num_classes), name='labels'),
        'node_subgraph': tf.placeholder(tf.int32, shape=(None), name='node_subgraph'),
        'dropout': tf.placeholder(DTYPE, shape=(None), name='dropout'),
        'adj_subgraph' : tf.sparse_placeholder(DTYPE,name='adj_subgraph',shape=(None,None)),
        'adj_subgraph_0' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_0'),
        'adj_subgraph_1' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_1'),
        'adj_subgraph_2' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_2'),
        'adj_subgraph_3' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_3'),
        'adj_subgraph_4' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_4'),
        'adj_subgraph_5' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_5'),
        'adj_subgraph_6' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_6'),
        'adj_subgraph_7' : tf.sparse_placeholder(DTYPE,name='adj_subgraph_7'),
        'dim0_adj_sub' : tf.placeholder(tf.int64,shape=(None),name='dim0_adj_sub'),
        'norm_loss': tf.placeholder(DTYPE,shape=(None),name='norm_loss'),
        'is_train': tf.placeholder(tf.bool, shape=(None), name='is_train'),
        'male_mask': tf.placeholder(tf.int32, shape=(None), name='male_mask'),
        'female_mask': tf.placeholder(tf.int32, shape=(None), name='female_mask')
    }
    return placeholders


#########
# TRAIN #
#########
def prepare(train_data,train_params,arch_gcn):
    adj_full,adj_train,feats,class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    male_users, female_users = load_male_female_users(FLAGS.data_prefix)

    placeholders = construct_placeholders(num_classes)
    minibatch = Minibatch(adj_full, adj_full_norm, adj_train, role, class_arr, placeholders, train_params, male_users, female_users)
    model = GraphSAINT(num_classes, placeholders,
                feats, arch_gcn, train_params, adj_full_norm, logging=True)

    # Initialize session
    sess = tf.Session(config=tf.ConfigProto(device_count={"CPU":40},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44,log_device_placement=FLAGS.log_device_placement))
    ph_misc_stat = {'val_f1_micro': tf.placeholder(DTYPE, shape=()),
                    'val_f1_macro': tf.placeholder(DTYPE, shape=()),
                    'train_f1_micro': tf.placeholder(DTYPE, shape=()),
                    'train_f1_macro': tf.placeholder(DTYPE, shape=()),
                    'time_per_epoch': tf.placeholder(DTYPE, shape=()),
                    'size_subgraph': tf.placeholder(tf.int32, shape=())}
    merged = tf.summary.merge_all()

    with tf.name_scope('summary'):
        _misc_val_f1_micro = tf.summary.scalar('val_f1_micro', ph_misc_stat['val_f1_micro'])
        _misc_val_f1_macro = tf.summary.scalar('val_f1_macro', ph_misc_stat['val_f1_macro'])
        _misc_train_f1_micro = tf.summary.scalar('train_f1_micro', ph_misc_stat['train_f1_micro'])
        _misc_train_f1_macro = tf.summary.scalar('train_f1_macro', ph_misc_stat['train_f1_macro'])
        _misc_time_per_epoch = tf.summary.scalar('time_per_epoch',ph_misc_stat['time_per_epoch'])
        _misc_size_subgraph = tf.summary.scalar('size_subgraph',ph_misc_stat['size_subgraph'])

    misc_stats = tf.summary.merge([_misc_val_f1_micro,_misc_val_f1_macro,_misc_train_f1_micro,_misc_train_f1_macro,
                    _misc_time_per_epoch,_misc_size_subgraph])
    summary_writer = tf.summary.FileWriter(log_dir(FLAGS.train_config,FLAGS.data_prefix,git_branch,git_rev,timestamp), sess.graph)
    # Init variables
    sess.run(tf.global_variables_initializer())
    return model,minibatch, sess, [merged,misc_stats],ph_misc_stat, summary_writer



def train(train_phases,arch_gcn,model,minibatch,\
            sess,train_stat,ph_misc_stat,summary_writer):
    
    if FLAGS.data_prefix == 'MovieLens':
        with open(FLAGS.thres_file, 'r') as fin:
            threshold = json.load(fin)
        # threshold = {}
        # threshold['dp'] = {'dev_low':0.1084,'dev_med':0.1626,'dev_high':0.2168,\
        #                   'test_low':0.0992,'test_med':0.1488,'test_high':0.1984}
        # threshold['eo'] = {'dev_low':0.0590,'dev_med':0.0884,'dev_high':0.1179,\
        #               '   test_low':0.0524,'test_med':0.0786,'test_high':0.1047}
    if FLAGS.fair:
        print('Incorporating Fairness Loss')
    else:
        print('No Incorporating Fairness Loss')
    print('outer_no {} inner_no {}'.format(FLAGS.outer_no,FLAGS.inner_no))
    
    print(threshold)


    avg_time = 0.0
    timing_steps = 0

    saver=tf.train.Saver()
    # timestamp = time.time()
    epoch_ph_start = 0
    mrr_best = {'dp_high':0,'dp_med':0,'dp_low':0,'eo_high':0,'eo_med':0,'eo_low':0}
    
    time_train = 0
    time_prepare = 0
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,report_tensor_allocations_upon_oom=True)
    run_metadata = tf.RunMetadata()
    many_runs_timeline=[]

    for ip,phase in enumerate(train_phases):
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        printf('START PHASE {:4d}'.format(ip),style='underline')
        if ip == 0:
            minibatch.set_gamma(0.0)
        elif ip == 1:
            minibatch.set_gamma(FLAGS.gamma)
        for e in range(epoch_ph_start,int(phase['end'])):
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle()
            l_encoder_loss_tr = list()
            l_disc_loss_tr = list()
            l_mrr_tr = list()
            l_disc_acc_tr = list()
            l_f1_weight_tr = list()
            l_size_subg = list()
            time_train_ep = 0
            time_prepare_ep = 0
            
            while not minibatch.end():
                t0 = time.time()
                feed_dict, labels = minibatch.feed_dict(mode='train')
                t1 = time.time()

                if FLAGS.timeline:      # profile the code with Tensorflow Timeline
                    _, _, encoder_loss_train, disc_loss_train, pred_train, disc_acc_train, dbg = sess.run([\
                            model.encoder_opt_op, model.disc_opt_op, model.encoder_loss, model.disc_loss, model.encoder_preds, model.disc_acc], feed_dict=feed_dict,
                            options=options, run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    many_runs_timeline.append(chrome_trace)
                else:
                    _, _, encoder_loss_train, disc_loss_train, pred_train, disc_acc_train, check_disc_loss = sess.run([\
                            model.encoder_opt_op, model.disc_opt_op, model.encoder_loss, model.disc_loss, model.encoder_preds, model.disc_acc, model.check_disc_loss], feed_dict=feed_dict,
                            options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
                # if disc_loss_train > 1:
                #     print('check_disc_loss',check_disc_loss)
                
                t2 = time.time()
                time_train_ep += t2-t1
                time_prepare_ep += t1-t0
                if not minibatch.batch_num % FLAGS.eval_train_every:
                    f1_weight = calc_f1(labels,pred_train)
                    mrr = mean_reciprocal_rank(labels,pred_train)
                    l_encoder_loss_tr.append(encoder_loss_train)
                    l_disc_loss_tr.append(disc_loss_train)
                    l_mrr_tr.append(mrr)
                    l_disc_acc_tr.append(disc_acc_train)
                    l_f1_weight_tr.append(f1_weight)
                    l_size_subg.append(minibatch.size_subgraph)

            
            time_train += time_train_ep
            time_prepare += time_prepare_ep
            
            if FLAGS.cpu_eval:      # Full batch evaluation using CPU
                saver.save(sess,'./tmp.chkpt')
                with tf.device('/cpu:0'):
                    sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
                    sess_cpu.run(tf.global_variables_initializer())
                    saver = tf.train.Saver()
                    saver.restore(sess_cpu, './tmp.chkpt')
                    sess_eval=sess_cpu
            else:
                sess_eval=sess
            
            loss_val, mrr_val, f1_weight_val, disc_acc_val, dp_val, eo_val = \
                evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='val')
            loss_test, mrr_test, f1_weight_test, disc_acc_test, dp_test, eo_test = \
                evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='test')
            printf(' TRAIN         : loss = {:.5f}\tmrr = {:.5f}\tw_mic = {:.5f}\tdisc_acc = {:.5f}\tdisc_loss = {:.5f}'.format(
                f_mean(l_encoder_loss_tr),f_mean(l_mrr_tr),f_mean(l_f1_weight_tr),f_mean(l_disc_acc_tr),f_mean(l_disc_loss_tr)))
            printf(' VALIDATION:     loss = {:.5f}\tmrr = {:.5f}\tw_mic = {:.5f}\tdisc_acc = {:.5f}\tdp = {:.5f}\teo = {:.5f}'.format(
                loss_val,mrr_val,f1_weight_val,disc_acc_val,dp_val,eo_val),style='yellow')
            
            # if ip == 1:
            for fair_level in ['low','high','med']:
                if (dp_val < threshold['dp']['dev_'+fair_level] \
                and dp_test < threshold['dp']['test_'+fair_level] \
                and mrr_val > mrr_best['dp_'+fair_level]) or e == 0:
                    mrr_best['dp_'+fair_level] = mrr_val
                    
                    if not os.path.exists(FLAGS.dir_log+'/models'):
                        os.makedirs(FLAGS.dir_log+'/models')
                    print('  Saving models dp {}...'.format(fair_level))
                    savepath = saver.save(sess, '{}/models/saved_model_dp_{}_{}.chkpt'.format(FLAGS.dir_log,fair_level,timestamp),write_meta_graph=False,write_state=False)
               
 
                if (eo_val < threshold['eo']['dev_'+fair_level] \
                and eo_test < threshold['eo']['test_'+fair_level] \
                and mrr_val > mrr_best['eo_'+fair_level]) or e == 0:
                    mrr_best['eo_'+fair_level] = mrr_val
                    
                    if not os.path.exists(FLAGS.dir_log+'/models'):
                        os.makedirs(FLAGS.dir_log+'/models')
                    print('  Saving models eo {}...'.format(fair_level))
                    savepath = saver.save(sess, '{}/models/saved_model_eo_{}_{}.chkpt'.format(FLAGS.dir_log,fair_level,timestamp),write_meta_graph=False,write_state=False)

            if FLAGS.tensorboard:
                misc_stat = sess.run([train_stat[1]],feed_dict={\
                                        ph_misc_stat['val_mrr']: mrr_val,
                                        ph_misc_stat['val_f1_weight']: f1_weight_val,
                                        ph_misc_stat['train_mrr']: f_mean(l_mrr_tr),
                                        ph_misc_stat['train_f1_weight']: f_mean(l_f1_weight_tr),
                                        ph_misc_stat['time_per_epoch']: time_train_ep+time_prepare_ep,
                                        ph_misc_stat['size_subgraph']: f_mean(l_size_subg)})
                # tensorboard visualization
                summary_writer.add_summary(_, e)
                summary_writer.add_summary(misc_stat[0], e)
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!",style='yellow')
    # timelines = TimeLiner()
    # for tl in many_runs_timeline:
    #     timelines.update_timeline(tl)
    # timelines.save('timeline.json')
    
    for fair_level in ['low','med','high']:
        saver.restore(sess_eval, '{}/models/saved_model_dp_{}_{}.chkpt'.format(FLAGS.dir_log,fair_level,timestamp))
        loss_val, mrr_val, f1_weight_val, disc_acc_val, dp_val, eo_val = \
                evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='val')
        loss_test, mrr_test, f1_weight_test, disc_acc_test, dp_test, eo_test = \
            evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='test')
        printf(' TRAIN (Ep avg): loss = {:.5f}\tmrr = {:.5f}\tw_mic = {:.5f}\tdisc_acc = {:.5f}\tdisc_loss = {:.5f}'.format(
            f_mean(l_encoder_loss_tr),f_mean(l_mrr_tr),f_mean(l_f1_weight_tr),f_mean(l_disc_acc_tr),f_mean(l_disc_loss_tr)))
        printf(' VALIDATION:     loss = {:.5f}\tmrr = {:.5f}\tw_mic = {:.5f}\tdisc_acc = {:.5f}\tdp = {:.5f}\teo = {:.5f}'.format(
            loss_val,mrr_val,f1_weight_val,disc_acc_val,dp_val,eo_val),style='yellow')
        
        printf('Total training time: {:6.2f} sec'.format(time_train),style='red')
        with open( join(FLAGS.dir_output,'adv_dp_{}_{}.txt'.format(fair_level,FLAGS.dataset)),'a') as fo:
            fo.write('gamma {:.2f} outer_no {:d} inner_no {:2d}\n'.format(
                FLAGS.gamma, FLAGS.outer_no, FLAGS.inner_no))
            fo.write('dev_mrr {:.5f} dev_dp {:.5f}\n'.format(mrr_val, dp_val))
            fo.write('test_mrr {:.5f} test_dp {:.5f}\n'.format(mrr_test, dp_test))

        saver.restore(sess_eval, '{}/models/saved_model_eo_{}_{}.chkpt'.format(FLAGS.dir_log,fair_level,timestamp))
        loss_val, mrr_val, f1_weight_val, disc_acc_val, dp_val, eo_val = \
                evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='val')
        loss_test, mrr_test, f1_weight_test, disc_acc_test, dp_test, eo_test = \
            evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='test')
        printf(' TRAIN (Ep avg): loss = {:.5f}\tmrr = {:.5f}\tw_mic = {:.5f}\tdisc_acc = {:.5f}\tdisc_loss = {:.5f}'.format(
            f_mean(l_encoder_loss_tr),f_mean(l_mrr_tr),f_mean(l_f1_weight_tr),f_mean(l_disc_acc_tr),f_mean(l_disc_loss_tr)))
        printf(' VALIDATION:     loss = {:.5f}\tmrr = {:.5f}\tw_mic = {:.5f}\tdisc_acc = {:.5f}\tdp = {:.5f}\teo = {:.5f}'.format(
            loss_val,mrr_val,f1_weight_val,disc_acc_val,dp_val,eo_val),style='yellow')
        
        printf('Total training time: {:6.2f} sec'.format(time_train),style='red')
        with open(join(FLAGS.dir_output,'eo_{}_{}.txt'.format(fair_level,FLAGS.dataset)),'a') as fo:
            fo.write('gamma {:.2f} outer_no {:d} inner_no {:2d}\n'.format(
                FLAGS.gamma, FLAGS.outer_no, FLAGS.inner_no))
            fo.write('dev_mrr {:.5f} dev_eo {:.5f}\n'.format(mrr_val, eo_val))
            fo.write('test_mrr {:.5f} test_eo {:.5f}\n'.format(mrr_test, eo_test))


    for fair_level in ['low','med','high']:
        command = 'rm {}/models/saved_model_dp_{}_{}.chkpt*'.format(FLAGS.dir_log,fair_level,timestamp)
        print(command)
        os.system(command)
        command = 'rm {}/models/saved_model_eo_{}_{}.chkpt*'.format(FLAGS.dir_log,fair_level,timestamp)
        print(command)
        os.system(command)
    return mrr_best


########
# MAIN #
########

def train_main(argv=None):
    train_params,train_phases,train_data,arch_gcn = parse_n_prepare(FLAGS)
    model,minibatch,sess,train_stat,ph_misc_stat,summary_writer = prepare(train_data,train_params,arch_gcn)
    ret = train(train_phases,arch_gcn,model,minibatch,sess,train_stat,ph_misc_stat,summary_writer)
    return ret


if __name__ == '__main__':
    tf.app.run(main=train_main)

