rootdir_zuco1 = "../eego_henry/dataset/zuco1/"  #  on local
rootdir_zuco2 = "../eego_henry/dataset/zuco2/"
result_dir = "class_results"
# subjects (subejcts starting with "Z" are from ZuCo 1, subjects starting with "Y" are from ZuCo 2)
#subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH
#subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL',
         #   'YTL', "ZDN", "ZPH", "ZJN", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]  # 'YMS', 'YRH', #'ZJS

subjects = ["ZPH", "ZKH", "ZKW", "ZAB", "ZDM", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZMG"]
# subjects = ["ZPH"]

# for running the experiments with previously extracted feature only one subject (from each dataset) is necessary
run_eeg_extraction = False
feature_dir = "../eeg_features/"

# ML task {sentiment-bin, sentiment-tri, reldetect}
class_task = 'sentiment-bin'
# ML model {lstm, cnn}
model = 'lstm'
# word embeddings {none, glove (300d), bert}
embedding_type = 'bert'

# features sets {'text_only' , 'eeg_raw', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma', 'combi_eeg_raw', 'eye_tracking', 'combi_eye_tracking'}
# sentence level features: {'combi_concat', 'sent_eeg_theta'}
# combined models: {'eeg_eye_tracking', 'eeg4'} 'binary' ?
feature_set = ['eeg_raw']
# hyper-parameters to test - general
lstm_dim = [256]
lstm_layers = [1]
dense_dim = [128]
dropout = [0.3]
batch_size = 60
epochs = 120
lr = [0.000001]

# hyper-parameters for the convolutional EEG-decoding component, only apply if model = 'cnn' is selected
inception_filters = [14]
inception_kernel_sizes = [[1,4,7]]
inception_pool_size = [3]
inception_dense_dim = [(128,16)]

# best params raw eeg:
eeg_lstm_dim = [64]
eeg_dense_dim = [64]
eeg_dropout = [0.1]

# other parameters
folds = 5
#random_seed_values = [13, 78, 22, 66, 42]
random_seed_values = [13, 22, 42]
validation_split = 0.1
patience = 30
min_delta = 0.0000001
data_percentage = 0#0.75
drop_classes = []#, 1, 4, 6, 3, 9, 2]

# only for Relation Detection:
rel_thresholds = [0.3]#, 0.5, 0.7]
