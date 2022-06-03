'''
    Start training with the selected dataset, network, and settings.
'''

from src.py.training_on_the_fly.call_job_3D import call_job_3D
from src.py.utils.mkdir import mkdir as mkdir
from src.py.utils.data_splitting import get_split_train_val_test_airway, get_split_train_val_test_vessel, get_split_train_val_test_initial

# Path
Data_Path = '/PythonCodes/data/Preprocessed_data/'  # example path
Root_Path = '/PythonCodes/label_refinement_3D/'   # example path

Airway_PROCESSED_DATA_PARENT = Data_Path + "/Airway_preprocessed/"
Airway_RESULTS_DIR = Root_Path + "/resources/results_airway/"
Airway_Model_DIR = Root_Path + "/resources/airway_model/"
Vessel_PROCESSED_DATA_PARENT = Data_Path + "/CTA_Vessel_preprocessed/"
Vessel_RESULTS_DIR = Root_Path + "/resources/results_vessel/"
Vessel_Model_DIR = Root_Path + "/resources/vessel_model/"

PATCH_SHAPE = (128, 128, 128)
PAD_SHAPE = (128, 128, 128) # only used for testing
BASE_FEATURE = 16

IS_GENERATE_ERRORS_LABELS = True

# choose one dataset
dataset_list = [
                'airway',
                # 'vessel',
                ]

data_seed_list = [
                1,
                ]

network_list = ['unet']

method_list = [
            # 'CNN',              # train UNet
            # 'Adv',              # train Adversarial learning for segmentation
            # 'DoubleUNet',       # train DoubleUNet
            # 'GAN',              # train GAN
            # 'LR',               # only train Label refinement network on CNN results. Not end-to-end.
            # 'LR_Syn',           # train LR network + Synthesized dataset
            'LR_Syn_GAN',         # proposed method
            ]

# the GAN model name used for LASN
G_number = [
            99,
           ]

# synthetic error parameters
airway_error_1 = 0.25
airway_error_2 = 1.0

# Save previews
Preview_train = True
Preview_val = False
Save_nii = True

# Load pretrained network directly?
Pretrain_main = False

Number_training_batches_per_epoch = 250
Number_validation_batches_per_epoch = 50
epoch_restart = 0

mode_list = [
            'train',
            'test',
            ]

# Only for generating training and valid initial results from CNN model
Train_Valid_Initial_Test = False
# Test_Train_keys = 'train'
Test_Train_keys = 'val'

job_description = {
                    'preview_train': Preview_train,
                    'preview_val': Preview_val,
                    'Patch_size': PATCH_SHAPE,
                    'Pad_size': PAD_SHAPE,
                    'task': 'seg',
                    'device': 'cuda',
                    'val_epoch': 1,
                    'vis_train': 1,
                    'batch_size': 1,
                    'dropout_rate': 0.1,
                    'base_features': BASE_FEATURE,
                    'latent_gan': int(BASE_FEATURE * PATCH_SHAPE[0]/(2*2*2*2) * PATCH_SHAPE[1]/(2*2*2*2) * PATCH_SHAPE[2]/(2*2*2*2)*8),
                    'latent_adv': int(BASE_FEATURE * PATCH_SHAPE[0]/(2*2*2*2) * PATCH_SHAPE[1]/(2*2*2*2) * PATCH_SHAPE[2]/(2*2*2*2)),
                    'lr': 1e-2,
                    'lr_G': 1e-2,
                    'lr_D': 1e-2,
                    'epoch': 200,
                    'epoch_G': 100,
                    'epoch_D': 50,
                    'Dim': '3D',
                    'pretrain_main': Pretrain_main,
                    'save_nii': Save_nii,
                    'train_valid_initial_test': Train_Valid_Initial_Test,
                    'test_train_keys': Test_Train_keys,
                    'is_generate_errors_labels': IS_GENERATE_ERRORS_LABELS,
                    'number_training_batches_per_epoch': Number_training_batches_per_epoch,
                    'number_validation_batches_per_epoch': Number_validation_batches_per_epoch,
                    'G_number': G_number[0],
                    'airway_error_1': airway_error_1,
                    'airway_error_2': airway_error_2,
                    'epoch_restart': epoch_restart,
                    }

for dataset in dataset_list:
    for method in method_list:
        for seed in data_seed_list:
            for network in network_list:
                for mode in mode_list:

                    if dataset == 'airway':
                        job_description.update({
                            'RESULTS_DIR': Airway_RESULTS_DIR,
                            'Patient_DIR': Airway_PROCESSED_DATA_PARENT,
                            'Model_DIR': Airway_Model_DIR,
                        })

                    elif dataset == 'vessel':
                        job_description.update({
                            'RESULTS_DIR': Vessel_RESULTS_DIR,
                            'Patient_DIR': Vessel_PROCESSED_DATA_PARENT,
                            'Model_DIR': Vessel_Model_DIR,
                        })

                    job_description.update({
                        'dataset': dataset,
                        'method': method,
                        'seed': seed,
                        'network': network,
                        'mode': mode,
                    })

                    if dataset == 'airway':
                        train, val, test, unlabel = get_split_train_val_test_airway(job_description)
                    elif dataset == 'vessel':
                        train, val, test = get_split_train_val_test_vessel(job_description)

                    patients_keys = train + val
                    job_description.update({
                        'patients_keys': patients_keys,
                        'train_keys': train,
                        'val_keys': val,
                        'test_keys': test,
                    })

                    if method == 'GAN' or method == 'LR' or method == 'LR_Syn' or method == 'LR_Syn_GAN':
                        train_initial, val_initial, test_initial, unlabel_initial = get_split_train_val_test_initial(job_description)
                        job_description.update({
                            'train_initial_keys': train_initial,
                            'val_initial_keys': val_initial,
                            'test_initial_keys': test_initial,
                        })

                    if method == 'LR_Syn_GAN_onlineError':
                        train_initial, val_initial, test_initial = get_split_train_val_test_initial(job_description)
                        job_description.update({
                            'train_initial_keys': train_initial,
                            'val_initial_keys': val_initial,
                            'test_initial_keys': test_initial,
                        })

                    basic_path = job_description['RESULTS_DIR']

                    if job_description['G_number'] == 'random_G':
                        result_path = basic_path + '/results_' + job_description['method'] + '/seed_' + str(
                            job_description['seed']) + '/[' + job_description['dataset'] + \
                                      ']_[' + job_description['network'] + ']_[' \
                                      + str(job_description['base_features']) + ']_' + '[seed_' \
                                      + str(job_description['seed']) + ']' + '_[' + str(PATCH_SHAPE[0]) + '_' + str(PAD_SHAPE[0]) + ']_' + 'random_G'
                                      
                    else:
                        result_path = basic_path + '/results_' + job_description['method'] + '/seed_' + str(
                            job_description['seed']) + '/[' + job_description['dataset'] + \
                                      ']_[' + job_description['network'] + ']_[' \
                                      + str(job_description['base_features']) + ']_' + '[seed_' \
                                      + str(job_description['seed']) + ']' + '_[' + str(PATCH_SHAPE[0]) + '_' + str(PAD_SHAPE[0]) + ']_' + str(G_number[0])

                    job_description['result_path'] = result_path
                    mkdir(result_path)

                    call_job_3D(job_description)

print(job_description['Patient_DIR'] + "finished!")
