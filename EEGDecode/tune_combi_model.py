import config
from feature_extraction import zuco_reader
from reldetect import reldetect_text_eeg_model, reldetect_eeg_gaze_model, reldetect_text_eeg4_model, reldetect_text_eeg_gaze_model, reldetect_text_random_model
from ner import ner_text_model
from sentiment import sentiment_eeg_model, sentiment_eeg_gaze_model, sentiment_text_eeg_gaze_model, sentiment_text_eeg4_model, sentiment_text_random_model, sentiment_text_eeg_model
from data_helpers import save_results, load_matlab_files
import numpy as np
import collections
import json
import sys
import os
import tensorflow as tf
import random
from datetime import timedelta
import time

def main():
    start = time.time()
    feature_dict = {}
    label_dict = {}
    eeg_dict = {}
    gaze_dict = {}
    print("TASK: ", config.class_task)
    print("Extracting", config.feature_set[0], "features....")
    for subject in config.subjects:
        loaded_data = load_matlab_files(config.class_task, subject)

        zuco_reader.extract_features(loaded_data, config.feature_set, feature_dict, eeg_dict, gaze_dict)
        zuco_reader.extract_labels(feature_dict, label_dict, config.class_task, subject)

        elapsed = (time.time() - start)
        print('{}: {}'.format(subject, timedelta(seconds=int(elapsed))))

    if config.run_eeg_extraction:
        # save EEG features
        with open(config.feature_dir + config.feature_set[0] + '_feats_file_'+config.class_task+'.json', 'w') as fp:
            json.dump(eeg_dict, fp)
        print("saved.")
        sys.exit()
    else:
        print("Reading gaze features from file!!")
        #gaze_dict = json.load(open("feature_extraction/features/gaze_feats_file_" + config.class_task + ".json"))

        print("Reading EEG features from file!!")
        if 'eeg4' in config.feature_set:
            eeg_dict_theta = json.load(open("../eeg_features/eeg_theta_feats_file_" + config.class_task + ".json"))
            eeg_dict_beta = json.load(open("../eeg_features/eeg_beta_feats_file_" + config.class_task + ".json"))
            eeg_dict_alpha = json.load(open("../eeg_features/eeg_alpha_feats_file_" + config.class_task + ".json"))
            eeg_dict_gamma = json.load(open("../eeg_features/eeg_gamma_feats_file_" + config.class_task + ".json"))
        elif 'text_eeg_eye_tracking' in config.feature_set:
            eeg_dict = json.load(open("../eeg_features/combi_eeg_raw_feats_file_"+ config.class_task + ".json"))
        else:
            eeg_dict = json.load(
            open("../eeg_features/" + config.feature_set[0] + "_feats_file_" + config.class_task + ".json"))

        print("done, ", len(eeg_dict), " sentences with EEG features.")

    

    feature_dict = collections.OrderedDict(sorted(feature_dict.items()))
    label_dict = collections.OrderedDict(sorted(label_dict.items()))
    
    eeg_dict = collections.OrderedDict(sorted(eeg_dict.items()))
    gaze_dict = collections.OrderedDict(sorted(gaze_dict.items()))

    if 'eeg4' in config.feature_set:
        eeg_dict_alpha = collections.OrderedDict(sorted(eeg_dict_alpha.items()))
        eeg_dict_beta = collections.OrderedDict(sorted(eeg_dict_beta.items()))
        eeg_dict_theta = collections.OrderedDict(sorted(eeg_dict_theta.items()))
        eeg_dict_gamma = collections.OrderedDict(sorted(eeg_dict_gamma.items()))

    print(len(feature_dict.keys()), len(label_dict))

    if 'eeg4' in config.feature_set:
        if len(set([len(feature_dict), len(label_dict), len(eeg_dict_alpha), len(eeg_dict_beta), len(eeg_dict_gamma), len(eeg_dict_theta)])) > 1:
            print("WARNING: Not an equal number of sentences in features and labels!")
        print('len(feature_dict):\t', len(feature_dict))
        print('len(label_dict):\t', len(label_dict))
        print('len(eeg_dict_alpha):\t', len(eeg_dict_alpha))
        print('len(eeg_dict_beta):\t', len(eeg_dict_beta))
        print('len(eeg_dict_gamma):\t', len(eeg_dict_gamma))
        print('len(eeg_dict_theta):\t', len(eeg_dict_theta))
    else:
        if len(feature_dict) != len(label_dict) or len(feature_dict) != len(eeg_dict) or len(label_dict) != len(eeg_dict):
            print("WARNING: Not an equal number of sentences in features and labels!")
        print('len(feature_dict): {}\nlen(label_dict): {}\nlen(eeg_dict): {}'.format(len(feature_dict), len(label_dict), len(eeg_dict)))
        if "eye_tracking" in config.feature_set[0]:
            print('len(feature_dict): {}'.format(len(gaze_dict)))

    print('Starting Loop')
    start = time.time()
    count = 0

    for rand in config.random_seed_values:
        np.random.seed(rand)
        tf.random.set_seed(rand)
        os.environ['PYTHONHASHSEED'] = str(rand)
        random.seed(rand)
        for lstmDim in config.lstm_dim:
            for lstmLayers in config.lstm_layers:
                for denseDim in config.dense_dim:
                    for drop in config.dropout:
                        for bs in config.batch_size:
                            for lr_val in config.lr:
                                for e_val in config.epochs:
                                    for inception_filters in config.inception_filters:
                                        for inception_kernel_sizes in config.inception_kernel_sizes:
                                            for inception_pool_size in config.inception_pool_size:
                                                for inception_dense_dim in config.inception_dense_dim:
                                                    parameter_dict = {"lr": lr_val, "lstm_dim": lstmDim, "lstm_layers": lstmLayers,
                                                                    "dense_dim": denseDim, "dropout": drop, "batch_size": bs,
                                                                    "epochs": e_val, "random_seed": rand, "inception_filters": inception_filters,
                                                                    "inception_dense_dim": inception_dense_dim, "inception_kernel_sizes": inception_kernel_sizes,
                                                                    "inception_pool_size": inception_pool_size}

                                                    if config.class_task == 'reldetect':
                                                        for threshold in config.rel_thresholds:
                                                            if 'eeg4' in config.feature_set:
                                                                fold_results = reldetect_text_eeg4_model.classifier(feature_dict,
                                                                                                                        label_dict,
                                                                                                                        eeg_dict_theta,
                                                                                                                        eeg_dict_alpha,
                                                                                                                        eeg_dict_beta,
                                                                                                                        eeg_dict_gamma,
                                                                                                                        config.embeddings,
                                                                                                                        parameter_dict,
                                                                                                                        rand, threshold)
                                                            elif 'random' in config.feature_set and 'eeg_theta' in config.feature_set:
                                                                fold_results = reldetect_text_random_model.classifier(feature_dict, label_dict,
                                                                                                                eeg_dict,
                                                                                                                config.embeddings,
                                                                                                                parameter_dict,
                                                                                                                rand, threshold)
                                                            elif 'combi_eeg_raw' in config.feature_set or 'eeg_theta' in config.feature_set or 'eeg_alpha' in config.feature_set or 'eeg_beta' in config.feature_set or 'eeg_gamma' in config.feature_set:
                                                                fold_results = reldetect_text_eeg_model.classifier(feature_dict,
                                                                                                                        label_dict,
                                                                                                                        eeg_dict,
                                                                                                                        config.embeddings,
                                                                                                                        parameter_dict,
                                                                                                                        rand, threshold)
                                                            elif 'eeg_eye_tracking' in config.feature_set:
                                                                fold_results = reldetect_eeg_gaze_model.classifier(label_dict,
                                                                                                                        eeg_dict,
                                                                                                                        gaze_dict,
                                                                                                                        config.embeddings,
                                                                                                                        parameter_dict,
                                                                                                                        rand, threshold)
                                                            elif 'text_eeg_eye_tracking' in config.feature_set:
                                                                print(config.feature_set)
                                                                fold_results = reldetect_text_eeg_gaze_model.classifier(feature_dict, label_dict,
                                                                                                                        eeg_dict,
                                                                                                                        gaze_dict,
                                                                                                                        config.embeddings,
                                                                                                                        parameter_dict,
                                                                                                                        rand, threshold)
                                                            save_results(fold_results, config.class_task)

                                                    elif config.class_task == 'sentiment-tri':
                                                        if 'eeg4' in config.feature_set:
                                                            fold_results = sentiment_text_eeg4_model.classifier(feature_dict, label_dict, eeg_dict_theta,
                                                                                                            eeg_dict_alpha, eeg_dict_beta, eeg_dict_gamma,
                                                                                                            config.embeddings,
                                                                                                            parameter_dict,
                                                                                                            rand)

                                                        if 'eeg_raw' in config.feature_set:
                                                            fold_results = sentiment_eeg_model.classifier(label_dict,
                                                                                                            eeg_dict,
                                                                                                            config.embeddings,
                                                                                                            parameter_dict,
                                                                                                            rand)
                                                        elif 'text_eeg_eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_text_eeg_gaze_model.classifier(feature_dict,
                                                                                                                        label_dict,
                                                                                                                        eeg_dict,
                                                                                                                        gaze_dict,
                                                                                                                        config.embeddings,
                                                                                                                        parameter_dict,
                                                                                                                        rand)
                                                        elif 'eeg_eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_eeg_gaze_model.classifier(label_dict,
                                                                                                                    eeg_dict, gaze_dict,
                                                                                                                    config.embeddings,
                                                                                                                    parameter_dict,
                                                                                                                    rand)

                                                        elif 'random' in config.feature_set and 'eeg_theta' in config.feature_set:
                                                            fold_results = sentiment_text_random_model.classifier(feature_dict, label_dict,
                                                                                                            eeg_dict,
                                                                                                            config.embeddings,
                                                                                                            parameter_dict,
                                                                                                            rand)

                                                        elif 'combi_eeg_raw' in config.feature_set or 'eeg_theta' in config.feature_set or 'eeg_alpha' in config.feature_set or 'eeg_beta' in config.feature_set or 'eeg_gamma' in config.feature_set:
                                                            fold_results = sentiment_text_eeg_model.classifier(feature_dict,
                                                                                                                label_dict,
                                                                                                                eeg_dict,
                                                                                                                config.embeddings,
                                                                                                                parameter_dict,
                                                                                                                rand)

                                                        save_results(fold_results, config.class_task)
                                                    elif config.class_task == 'sentiment-bin':
                                                        print("dropping neutral sentences for binary sentiment classification")
                                                        for s, label in list(label_dict.items()):
                                                            # drop neutral sentences for binary sentiment classification
                                                            if label == 2:
                                                                del label_dict[s]
                                                                del feature_dict[s]
                                                                if 'eeg4' in config.feature_set:
                                                                    del eeg_dict_theta[s]
                                                                    del eeg_dict_alpha[s]
                                                                    del eeg_dict_beta[s]
                                                                    del eeg_dict_gamma[s]
                                                                else:
                                                                    del eeg_dict[s]

                                                        if 'eeg4' in config.feature_set:
                                                            fold_results = sentiment_text_eeg4_model.classifier(feature_dict, label_dict, eeg_dict_theta,
                                                                                                            eeg_dict_alpha, eeg_dict_beta, eeg_dict_gamma,
                                                                                                            config.embeddings,
                                                                                                            parameter_dict,
                                                                                                            rand)

                                                        if 'eeg_raw' in config.feature_set:
                                                            fold_results = sentiment_eeg_model.classifier(label_dict,
                                                                                                            eeg_dict,
                                                                                                            config.embeddings,
                                                                                                            parameter_dict,
                                                                                                            rand)
                                                        elif 'text_eeg_eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_text_eeg_gaze_model.classifier(feature_dict,
                                                                                                                        label_dict,
                                                                                                                        eeg_dict,
                                                                                                                        gaze_dict,
                                                                                                                        config.embeddings,
                                                                                                                        parameter_dict,
                                                                                                                        rand)
                                                        elif 'eeg_eye_tracking' in config.feature_set:
                                                            fold_results = sentiment_eeg_gaze_model.classifier(label_dict,
                                                                                                                    eeg_dict, gaze_dict,
                                                                                                                    config.embeddings,
                                                                                                                    parameter_dict,
                                                                                                                    rand)

                                                        elif 'random' in config.feature_set and 'eeg_theta' in config.feature_set:
                                                            fold_results = sentiment_text_random_model.classifier(feature_dict, label_dict,
                                                                                                            eeg_dict,
                                                                                                            config.embeddings,
                                                                                                            parameter_dict,
                                                                                                            rand)

                                                        elif 'combi_eeg_raw' in config.feature_set or 'eeg_theta' in config.feature_set or 'eeg_alpha' in config.feature_set or 'eeg_beta' in config.feature_set or 'eeg_gamma' in config.feature_set:
                                                            fold_results = sentiment_text_eeg_model.classifier(feature_dict,
                                                                                                                label_dict,
                                                                                                                eeg_dict,
                                                                                                                config.embeddings,
                                                                                                                parameter_dict,
                                                                                                                rand)

                                                        save_results(fold_results, config.class_task)

                                                    elapsed = (time.time() - start)
                                                    print('iteration {} done'.format(count))
                                                    print('Time since starting the loop: {}'.format(timedelta(seconds=int(elapsed))))
                                                    count += 1

if __name__ == '__main__':
    main()
