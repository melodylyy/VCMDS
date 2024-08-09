import numpy as np
from data_preprocessing import *
from sklearn.model_selection import KFold
from utilize import *
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from model import  GCNModelVAE
import scipy.sparse as sp
import tensorflow as tf
from optimizer import  OptimizerVAE
import time
from Get_low_dimension_feature import get_low_feature
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 48, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2',8, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'miRNA-disease', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Number of features.')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
# Parameters
'''k1 = 50
k2 = 20'''
D = 85 # MF dimension
A = np.loadtxt("dataset/guanlianmatrix.txt")

# Get the samples for all the positive and negative ones
samples = get_all_the_samples(A)
ii = 0
label_all = []
y_score_all = []
w = 0.3
# Cross validation
kf = KFold(n_splits=5, shuffle=True)
iter = 0  # Control each iterator
sum_score = 0
sum_score = 0
sum_AUPR = 0
sum_Accuracy= 0
sum_Precision = 0
sum_Recall = 0
sum_F1Score = 0
for train_index, test_index in kf.split(samples):
    if iter < 11:
        iter += 1
        train_samples = samples[train_index, :]
        test_samples = samples[test_index, :]
        new_A = update_Adjacency_matrix(A, test_samples)
        print(np.sum(new_A))
        sim_m, sim_d = get_syn_sim1(A,w)
        sim_m_0 = set_digo_zero(sim_m, 0)
        sim_d_0 = set_digo_zero(sim_d, 0)
        features_m = A
        features_d = A.transpose()
        features_m = sp.coo_matrix(features_m)
        adj_norm = preprocess_graph(sim_m_0)
        tf.disable_eager_execution()
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        num_nodes = sim_m.shape[0]

        features = sparse_to_tuple(features_m.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
        pos_weight = 1
        norm = 0.5

        opt = OptimizerVAE(preds=model.reconstructions,
                                labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                model=model, num_nodes=num_nodes,
                                   #pos_weight=pos_weight,
                                norm=norm)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cost_val = []
        acc_val = []
        val_roc_score = []

        sim_m_0 = sp.coo_matrix(sim_m_0)
        sim_m_0.eliminate_zeros()
        adj_label = sim_m_0 + sp.eye(sim_m_0.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

        # Getting the feature vectors for miRNAs
        Z = sess.run(model.z, feed_dict=feed_dict)
        feature_m = Z

        # Training disease by VGAE
        # Getting the feature extracting by VGAE on miRNA similarity network
        features_d = sp.coo_matrix(features_d)
        if FLAGS.features == 0:
            features = sparse_to_tuple(features_m.tocoo())
            num_features = features[2][1]
            features_nonzero = features[1].shape[0]


            features = sp.identity(num_features)

        adj_norm = preprocess_graph(sim_d_0)
        num_nodes = sim_d.shape[0]

        features = sparse_to_tuple(features_d.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        # Create model
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)


        pos_weight = 1
        norm = 0.5

        # Optimizer
        opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=model, num_nodes=num_nodes,
                                   #pos_weight=pos_weight,
                                   norm=norm)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cost_val = []
        acc_val = []
        val_roc_score = []

        sim_d_0 = sp.coo_matrix(sim_d_0)
        sim_d_0.eliminate_zeros()
        adj_label = sim_d_0 + sp.eye(sim_d_0.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

        Z = sess.run(model.z, feed_dict=feed_dict)
        Z =np.array(Z)

        feature_d = Z

        feature_MFm, feature_MFd = get_low_feature(D,0.01, A, sim_m, sim_d,0.01,0.5, 0.01)

        #emerge the miRNA feature and disease feature
        vect_len1 = feature_m.shape[1]
        vect_len2 = feature_d.shape[1]
        train_n = train_samples.shape[0]
        train_feature = np.zeros([train_n, 2*vect_len1+2*D])
        train_label = np.zeros([train_n])
        for i in range(train_n):
            train_feature[i,0:vect_len1] = feature_m[train_samples[i,0],:]
            train_feature[i,vect_len1 :(vect_len1+vect_len2)] = feature_d[train_samples[i,1], :]
            train_feature[i,(vect_len1+vect_len2):(vect_len1+vect_len2+D)] = feature_MFm[train_samples[i,0],:]
            train_feature[i, (vect_len1+vect_len2+D):(vect_len1+vect_len2+2*D)] = feature_MFd[train_samples[i,1],:]
            train_label[i] = train_samples[i,2]

        test_N = test_samples.shape[0]
        test_feature = np.zeros([test_N,2*vect_len1+2*D])

        test_label = np.zeros(test_N)
        for i in range(test_N):
            test_feature[i, 0:vect_len1] = feature_m[test_samples[i,0], :]
            test_feature[i, vect_len1:(vect_len1+vect_len2)] =  feature_d[test_samples[i,1], :]
            test_feature[i, (vect_len1+vect_len2):(vect_len1+vect_len2+D)] = feature_MFm[test_samples[i,0],:]
            test_feature[i, (vect_len1+vect_len2+D):(vect_len1+vect_len2+2*D)] = feature_MFd[test_samples[i,1], :]
            test_label[i]=test_samples[i,2]


        model = BuildModel(train_feature, train_label)



