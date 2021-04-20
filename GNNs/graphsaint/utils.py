import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml
import scipy.sparse as sp
from graphsaint.globals import *


def load_data(prefix, normalize=True):
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)

    adj_train = scipy.sparse.load_npz('./{}/split_{}/adj_train_{}.npz'.format(prefix,FLAGS.outer_no,FLAGS.inner_no)).astype(np.bool)
    role = json.load(open('./{}/split_{}/role_{}.json'.format(prefix,FLAGS.outer_no,FLAGS.inner_no)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


def load_male_female_users(prefix):
    male_users = json.load(open('./{}/male_users.json'.format(prefix)))
    female_users = json.load(open('./{}/female_users.json'.format(prefix)))
    
    male_users = [int(u) for u in male_users]
    female_users = [int(u) for u in female_users]

    male_users = np.asarray(male_users)
    female_users = np.asarray(female_users)
    
    return male_users, female_users

def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    INPUT:
        G           graph-tool graph, full graph including training,val,testing
        feats       ndarray of shape |V|xf
        class_map   dictionary {vertex_id: class_id}
        val_nodes   index of validation nodes
        test_nodes  index of testing nodes
    OUTPUT:
        G           graph-tool graph unchanged
        role        array of size |V|, indicating 'train'/'val'/'test'
        class_arr   array of |V|x|C|, converted by class_map
        feats       array of features unchanged
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role


def parse_layer_yml(arch_gcn,dim_input):
    num_layers = len(arch_gcn['arch'].split('-'))
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']]*num_layers
    act_layer = [arch_gcn['act']]*num_layers
    aggr_layer = [arch_gcn['aggr']]*num_layers
    dims_layer = [arch_gcn['dim']]*num_layers
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    return [dim_input]+dims_layer,order_layer,act_layer,bias_layer,aggr_layer





def parse_n_prepare(flags):
    with open(flags.train_config) as f_train_config:
        train_config = yaml.load(f_train_config)
    arch_gcn = {'dim':-1,'aggr':'concat','loss':'softmax','arch':'1','act':'I','bias':'norm'}
    arch_gcn.update(train_config['network'][0])
    train_params = {'lr':0.01,'weight_decay':0.,'norm_loss':True,'norm_aggr':True,'q_threshold':50,'q_offset':0}
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data(flags.data_prefix)
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")
    return train_params,train_phases,train_data,arch_gcn





def log_dir(f_train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = FLAGS.dir_log+"/tf_log/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir

def sess_dir(dims,train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = "saved_models/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            layer='-'.join(dims),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return sess_dir


def adj_norm(adj):
    """
    Normalize adj according to two methods: symmetric normalization and rw normalization.
    sym norm is used in the original GCN paper (kipf)
    rw norm is used in graphsage and some other variants.

    # Procedure: 
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by (D')^(-1/2) x adj' x (D')^(-1/2)
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten()
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    return adj_norm




##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg,style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))


