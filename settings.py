from srblib import abs_path
import os
data_dir = abs_path('./datasets')
file_name = os.path.join(data_dir, 'Eluvio_DS_Challenge.csv')
n_train_samples = 254618
n_val_samples = 127309
n_test_samples = 127309


# file_name = abs_path('./datasets/temp_data.csv')
# n_train_samples = 14
# n_val_samples = 7
# n_test_samples = 7