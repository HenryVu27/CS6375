import config
from feature_extraction import zuco_reader
from data_helpers import save_results, load_matlab_files
import numpy as np
import collections
import torch
import json
import sys
import ml_helpers
from datetime import timedelta
import time
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data():
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
        gaze_dict = json.load(open("feature_extraction/features/gaze_feats_file_" + config.class_task + ".json"))

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
    
    if 'eeg4' in config.feature_set:
        return feature_dict,label_dict,eeg_dict, gaze_dict, eeg_dict_alpha, eeg_dict_beta, eeg_dict_theta, eeg_dict_gamma
    else:
        return feature_dict,label_dict,eeg_dict, gaze_dict, None, None, None, None

def dict2features():
    feature_dict,label_dict, eeg_dict, gaze_dict, eeg_dict_alpha, eeg_dict_beta, eeg_dict_theta, eeg_dict_gamma =prepare_data()

    X_text = list(feature_dict.keys())
    y = list(label_dict.values())

    # check order of sentences in labels and features dicts
    if 'eeg4' in config.feature_set:
        if list(label_dict.keys())[0] != list(eeg_dict_alpha.keys())[0] != list(feature_dict.keys())[0] \
        != list(eeg_dict_beta.keys())[0] != list(eeg_dict_gamma.keys())[0] != list(eeg_dict_theta.keys())[0]:
            sys.exit("STOP! Order of sentences in labels and features dicts not the same!")
    else:
        if list(label_dict.keys())[0] != list(eeg_dict.keys())[0]:
            sys.exit("STOP! Order of sentences in labels and features dicts not the same!")
    
    # these are already one hot categorical encodings
    y = y

    # prepare text samples
    X_data_text, X_text_masks = ml_helpers.prepare_text(X_text, config.embedding_type)

    # prepare EEG data
    eeg_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(eeg_dict)
    eeg_X = ml_helpers.scale_feature_values(eeg_X)
    X_data_eeg = ml_helpers.pad_cognitive_feature_seqs(eeg_X, max_length_cogni, "eeg")

    return X_data_text, X_text_masks, X_data_eeg, y



class TextEEGDataset(Dataset):
    def __init__(self, text_features, text_masks, eeg_features, labels):
        self.text_features = text_features
        self.text_masks= text_masks
        self.eeg_features = eeg_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        return torch.tensor(self.text_features[idx]).to(device), \
            torch.tensor(self.text_masks[idx]).to(device), \
            torch.tensor(self.eeg_features[idx]).to(device), torch.tensor(self.labels[idx]).to(device)

