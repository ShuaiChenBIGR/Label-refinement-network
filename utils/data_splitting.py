import glob

from sklearn.model_selection import KFold
import numpy as np


def get_split_train_val(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys


def get_split_train_val_test_airway(job_description):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """

    train_keys_sorted = sorted(glob.glob(glob.glob(job_description['Patient_DIR'] + '/All')[0] + '/*.npy'))
    train_keys = [k[:-4] for k in train_keys_sorted]

    valid_keys_sorted = sorted(glob.glob(glob.glob(job_description['Patient_DIR'] + '/ValidData_4imgs')[0] + '/*.npy'))
    valid_keys = [k[:-4] for k in valid_keys_sorted]

    test_keys_sorted = sorted(glob.glob(glob.glob(job_description['Patient_DIR'] + '/TestData_41imgs')[0] + '/*.npy'))
    test_keys = [k[:-4] for k in test_keys_sorted]

    if job_description['method'] == 'LR_Syn_GAN_semi' or job_description['method'] == 'CNN' or job_description['method'] == 'LR_Syn_GAN' or job_description['method'] == 'LR_Syn':
        unlabeled_keys_sorted = sorted(
            glob.glob(glob.glob(job_description['Patient_DIR'] + '/UnlabeledData_100imgs')[0] + '/*.npy'))
        unlabeled_keys = [k[:-4] for k in unlabeled_keys_sorted]

    data_seed = job_description['seed']
    np.random.seed(data_seed)

    np.random.shuffle(train_keys)
    np.random.shuffle(unlabeled_keys)

    import time
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)

    if job_description['method'] == 'LR_Syn_GAN_semi' or job_description['method'] == 'CNN' or job_description['method'] == 'LR_Syn_GAN' or job_description['method'] == 'LR_Syn':
        return train_keys, valid_keys, test_keys, unlabeled_keys

    return train_keys, valid_keys, test_keys


def get_split_train_val_test_initial(job_description):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """

    train_keys_sorted = sorted(glob.glob(glob.glob(job_description['Patient_DIR'] + '/TrainData_initial')[0] + '/*.npy'))
    train_keys = [k[:-4] for k in train_keys_sorted]

    valid_keys_sorted = sorted(glob.glob(glob.glob(job_description['Patient_DIR'] + '/ValidData_initial')[0] + '/*.npy'))
    valid_keys = [k[:-4] for k in valid_keys_sorted]

    test_keys_sorted = sorted(glob.glob(glob.glob(job_description['Patient_DIR'] + '/TestData_initial')[0] + '/*.npy'))
    test_keys = [k[:-4] for k in test_keys_sorted]

    if job_description['method'] == 'LR_Syn_GAN_semi' or job_description['method'] == 'CNN' or job_description['method'] == 'LR_Syn_GAN' or job_description['method'] == 'LR_Syn':
        unlabeled_keys_sorted = sorted(
            glob.glob(glob.glob(job_description['Patient_DIR'] + '/UnlabeledData_initial')[0] + '/*.npy'))
        unlabeled_keys = [k[:-4] for k in unlabeled_keys_sorted]

    data_seed = job_description['seed']
    np.random.seed(data_seed)

    np.random.shuffle(train_keys)
    np.random.shuffle(unlabeled_keys)

    import time
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)

    if job_description['method'] == 'LR_Syn_GAN_semi' or job_description['method'] == 'CNN' or job_description['method'] == 'LR_Syn_GAN' or job_description['method'] == 'LR_Syn':
        return train_keys, valid_keys, test_keys, unlabeled_keys

    return train_keys, valid_keys, test_keys