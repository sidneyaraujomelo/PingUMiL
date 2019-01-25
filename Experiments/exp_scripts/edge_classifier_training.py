from __future__ import print_function
import json
import numpy as np
import random
import os

from networkx.readwrite import json_graph
from argparse import ArgumentParser
from sklearn.externals import joblib

import sys

def get_class_labels(labels, ids):
    return [labels[str(i)].index(1) for i in ids]

def listAverage(l):
    return sum(l)/float(len(l))

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(29)
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    dummy = DummyClassifier()
    dummy.fit(train_embeds, train_labels)
    all_f1 = []
    all_precision = []
    all_recall = []
    for i in range(5):
        #clf = SVC()
        #clf = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=500)
        #clf = MLPClassifier(max_iter=500)
        #clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500)
        #clf = MLPClassifier(hidden_layer_sizes=(512,512,512,512,512), max_iter=500)
        clf = MLPClassifier(hidden_layer_sizes=tuple(256 for x in range(10)), max_iter=500)
        #clf = SGDClassifier(tol=1e-4)
        #clf = DecisionTreeClassifier()
        #log.fit(train_embeds, train_labels)
        c = list(zip(train_labels, train_embeds))
        random.shuffle(c)
        train_labels_t, train_embeds_t = list(zip(*c))
        train_labels = list(train_labels_t)
        train_embeds = list(train_embeds_t)
        clf.fit(train_embeds, train_labels)
        all_f1.append(f1_score(test_labels, clf.predict(test_embeds)))
        all_precision.append(precision_score(test_labels, clf.predict(test_embeds)))
        all_recall.append(recall_score(test_labels, clf.predict(test_embeds)))
        #if (f1 > best_f1):
            #best_f1, best_precision, best_recall = f1, precision, recall
    #print("train_precision: {}".format(precision_score(train_labels, clf.predict(train_embeds))))
    return all_f1, all_precision, all_recall

if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on encoded edge data for edge prediction.")
    parser.add_argument("prefix_file", help="Prefix to the files of the dataset.")
    parser.add_argument("dataset_dir", help="Path to directory containing the dataset.")
    parser.add_argument("embed_dir", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("setting", help="Either val or test.")
    parser.add_argument("clf_folds_dir", help="Folder containing clf folds")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    data_dir = args.embed_dir 
    setting = args.setting
    prefix = args.prefix_file
    clf_dir = args.clf_folds_dir

    random.seed()

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/"+prefix+"-G.json")))
    labels = json.load(open(dataset_dir + "/"+prefix+"-class_map.json"))
    print("Loading folds...")
    folds_files = [f for f in os.listdir(clf_dir)]
    train_folds_files = sorted(folds_files)[1::2]
    test_folds_files = sorted(folds_files)[::2]
    print("Train folds files: {}".format(train_folds_files))
    print("Test folds files: {}".format(test_folds_files))
    train_folds = [json.load(open(os.path.join(clf_dir, f))) for f in train_folds_files]
    test_folds = [json.load(open(os.path.join(clf_dir, f))) for f in test_folds_files]    

    #read embeddings
    embeds = np.load(data_dir + "/val.npy")
    id_map = {}
    with open(data_dir + "/val.txt") as fp:
        for i, line in enumerate(fp):
            id_map[line.strip()] = i

    #train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    #test_ids = [n for n in G.nodes() if G.node[n][setting]]
    #train_labels = get_class_labels(labels, train_ids)
    #test_labels = get_class_labels(labels, test_ids)
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(len(train_folds)):
        print("Training fold {}".format(i))
        train_labels = []
        test_labels = []
        train_embeds = []
        test_embeds = []

        npos = 0
        nneg = 0
        tpos = 0
        tneg = 0     
    
        for edge in train_folds[i]:
             a = edge["source"]
             b = edge["target"]
             embed_edge = embeds[id_map[str(a)]]+embeds[id_map[str(b)]]
             #embed_edge = np.multiply(embeds[id_map[str(a)]],embeds[id_map[str(b)]])
             train_labels.append(edge["class"])
             train_embeds.append(embed_edge)
             if (edge["class"]==1):
                 npos = npos+1
             else:
                 nneg = nneg+1

        for edge in test_folds[i]:
             a = edge["source"]
             b = edge["target"]
             embed_edge = embeds[id_map[str(a)]]+embeds[id_map[str(b)]]
             #embed_edge = np.multiply(embeds[id_map[str(a)]],embeds[id_map[str(b)]])
             test_labels.append(edge["class"])
             test_embeds.append(embed_edge)
             if (edge["class"]==1):
                 tpos = tpos+1
             else:
                 tneg = tneg+1

        print("N of Train Edges "+str(len(train_embeds)))
        print("N of Test Edges "+str(len(test_embeds)))
        print("N of positive and negative train edges "+str(npos)+" "+str(nneg))
        print("N of positive and negative test edges {} {}".format(tpos, tneg))
        print("Running regression..")
        f1, precision, recall  = run_regression(train_embeds, train_labels, test_embeds, test_labels)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print("Precision Recall F1")
    for i in range(len(precision_list)):
        print("Fold {}".format(i))
        print("Mean {} {} {}".format(np.mean(precision_list[i]), np.mean(recall_list[i]), np.mean(f1_list[i])))
        print("Variance {} {} {}".format(np.var(precision_list[i]), np.var(recall_list[i]), np.var(f1_list[i])))
